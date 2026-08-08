#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from medrag_repro.config import load_config
from medrag_repro.datamodels import CorpusDoc, PoisonDoc, QAItem
from medrag_repro.defense.drs import DRSDetector
from medrag_repro.evaluation.rag_eval import evaluate_attack
from medrag_repro.llm.client import load_openai_client
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import load_index, retrieve_topk
from medrag_repro.utils.io import read_jsonl, write_json, write_jsonl
from medrag_repro.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    client = load_openai_client()
    clean_queries = [QAItem(**row) for row in read_jsonl(cfg["paths"]["clean_queries"])]
    targets = [QAItem(**row) for row in read_jsonl(cfg["paths"]["targets"])]
    corpus = [CorpusDoc(**row) for row in read_jsonl(cfg["paths"]["pubmed_corpus"])]
    poison = [PoisonDoc(**row) for row in read_jsonl(cfg["paths"]["poison_docs"])]
    doc_lookup = {d.doc_id: d.text for d in corpus}
    index, doc_ids, vectors = load_index(cfg["paths"]["index_dir"])
    encoder = ContrieverEncoder(
        model_name=cfg["retriever"]["model_name"],
        batch_size=cfg["retriever"]["batch_size"],
        device=cfg["retriever"]["device"],
    )

    clean_ref_map = {}
    for qa in clean_queries:
        retrieved = retrieve_topk(qa.question, encoder, index, doc_ids, doc_lookup, cfg["retriever"]["top_k"])
        for doc_id, _, text in retrieved:
            clean_ref_map.setdefault(doc_id, text)
    clean_ref_texts = list(clean_ref_map.values())

    detector = DRSDetector(
        encoder=encoder,
        M=cfg["drs"]["M"],
        clean_quantile=cfg["drs"]["clean_threshold_quantile"],
    )
    detector.fit(clean_ref_texts)

    poison_texts = [p.full_text for p in poison]
    poison_scores = detector.score_texts(poison_texts)
    detected = detector.detect(poison_texts)
    kept_poison = [p for p, d in zip(poison, detected) if not d]

    clean_scores = detector.clean_scores
    threshold = detector.upper_threshold

    post = evaluate_attack(
        client=client,
        model=cfg["llm_eval"]["answer_model"],
        encoder=encoder,
        index=index,
        doc_ids=doc_ids,
        vectors=vectors,
        doc_lookup=doc_lookup,
        targets=targets,
        poison_docs=kept_poison,
        k=cfg["retriever"]["top_k"],
        answer_temperature=cfg["llm_eval"]["answer_temperature"],
    )

    details = [
        {"poison_id": p.poison_id, "target_qid": p.target_qid, "score": float(s), "detected": bool(d)}
        for p, s, d in zip(poison, poison_scores.tolist(), detected)
    ]
    out = {
        "n_clean_reference_docs": len(clean_ref_texts),
        "threshold": threshold,
        "empirical_clean_fpr": float(np.mean(clean_scores > threshold)),
        "poison_detection_rate": float(np.mean(detected)) if detected else 0.0,
        "n_poison_before": len(poison),
        "n_poison_after": len(kept_poison),
        "post_defense_attack_success_rate": post["attack_success_rate"],
        "post_defense_retrieval_precision": post["retrieval_precision"],
        "post_defense_retrieval_recall": post["retrieval_recall"],
        "post_defense_retrieval_f1": post["retrieval_f1"],
        "details": details,
    }
    write_json(cfg["paths"]["drs_metrics"], out)
    write_jsonl(str(cfg["paths"]["artifact_dir"]) + "/kept_poison_after_drs.jsonl", kept_poison)
    print({k: v for k, v in out.items() if k != "details"})


if __name__ == "__main__":
    main()
