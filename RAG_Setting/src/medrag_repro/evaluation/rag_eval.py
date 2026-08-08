from __future__ import annotations

import re
from typing import Dict, List, Sequence

import numpy as np
from openai import OpenAI

from medrag_repro.datamodels import PoisonDoc, QAItem
from medrag_repro.llm.client import chat_completion
from medrag_repro.llm.prompts import (
    answer_with_context_system_prompt,
    answer_with_context_user_prompt,
)
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import retrieve_topk


def answer_medqa_with_context(
    client: OpenAI,
    model: str,
    qa: QAItem,
    contexts: Sequence[str],
    temperature: float = 0.0,
) -> str:
    out = chat_completion(
        client,
        model,
        answer_with_context_system_prompt(),
        answer_with_context_user_prompt(contexts, qa.question, qa.options),
        temperature=temperature,
        max_tokens=50,
    )
    m = re.search(r'"predicted_option"\s*:\s*"?([A-Z])"?', out)
    return m.group(1) if m else ""


def add_poison_to_index(
    index: dict,
    doc_ids: List[str],
    vectors: np.ndarray,
    poison: Sequence[PoisonDoc],
    encoder: ContrieverEncoder,
) -> tuple[dict, List[str], np.ndarray]:
    poison_texts = [p.full_text for p in poison]
    poison_vecs = encoder.encode(poison_texts)

    new_doc_ids = doc_ids + [p.poison_id for p in poison]
    new_vectors = np.vstack([vectors, poison_vecs])

    new_index = {
        "backend": index.get("backend", "numpy"),
        "vectors": new_vectors,
    }

    return new_index, new_doc_ids, new_vectors


def evaluate_attack(
    client: OpenAI,
    model: str,
    encoder: ContrieverEncoder,
    index: dict,
    doc_ids: List[str],
    vectors: np.ndarray,
    doc_lookup: Dict[str, str],
    targets: Sequence[QAItem],
    poison_docs: Sequence[PoisonDoc],
    k: int,
    answer_temperature: float = 0.0,
) -> dict:
    grouped_poison: Dict[str, set] = {}
    for p in poison_docs:
        grouped_poison.setdefault(p.target_qid, set()).add(p.poison_id)

    if poison_docs:
        eval_index, eval_doc_ids, _ = add_poison_to_index(
            index, doc_ids, vectors, poison_docs, encoder
        )
    else:
        eval_index, eval_doc_ids, _ = index, doc_ids, vectors

    eval_doc_lookup = dict(doc_lookup)
    eval_doc_lookup.update({p.poison_id: p.full_text for p in poison_docs})

    retrieval_hits = 0
    retrieval_total = len(targets) * k
    target_recall_hits = 0
    target_recall_total = len(poison_docs)
    asr_hits = 0
    details = []

    for qa in targets:
        retrieved = retrieve_topk(
            qa.question, encoder, eval_index, eval_doc_ids, eval_doc_lookup, k
        )
        retrieved_ids = [x[0] for x in retrieved]
        retrieved_texts = [x[2] for x in retrieved]

        poison_ids_for_q = grouped_poison.get(qa.qid, set())
        hit_count = sum(1 for doc_id in retrieved_ids if doc_id in poison_ids_for_q)

        retrieval_hits += hit_count
        target_recall_hits += hit_count

        pred = answer_medqa_with_context(
            client, model, qa, retrieved_texts, answer_temperature
        )
        attack_success = pred == qa.target_option
        if attack_success:
            asr_hits += 1

        details.append(
            {
                "qid": qa.qid,
                "predicted_option": pred,
                "target_option": qa.target_option,
                "correct_option": qa.correct_option,
                "attack_success": attack_success,
                "retrieved_ids": retrieved_ids,
                "poison_hit_count": hit_count,
            }
        )

    precision = retrieval_hits / max(retrieval_total, 1)
    recall = target_recall_hits / max(target_recall_total, 1)
    f1 = 2 * precision * recall / max((precision + recall), 1e-12)
    asr = asr_hits / max(len(targets), 1)

    return {
        "n_targets": len(targets),
        "n_poison_docs": len(poison_docs),
        "k": k,
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "retrieval_f1": f1,
        "attack_success_rate": asr,
        "details": details,
    }