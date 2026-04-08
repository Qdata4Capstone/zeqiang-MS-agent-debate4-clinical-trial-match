#!/usr/bin/env python3
from __future__ import annotations

import argparse

from medrag_repro.config import load_config
from medrag_repro.datamodels import CorpusDoc, PoisonDoc, QAItem
from medrag_repro.evaluation.rag_eval import evaluate_attack
from medrag_repro.llm.client import load_openai_client
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import load_index
from medrag_repro.utils.io import read_jsonl, write_json
from medrag_repro.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    client = load_openai_client()
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

    out = evaluate_attack(
        client=client,
        model=cfg["llm_eval"]["answer_model"],
        encoder=encoder,
        index=index,
        doc_ids=doc_ids,
        vectors=vectors,
        doc_lookup=doc_lookup,
        targets=targets,
        poison_docs=poison,
        k=cfg["retriever"]["top_k"],
        answer_temperature=cfg["llm_eval"]["answer_temperature"],
    )
    write_json(cfg["paths"]["attack_metrics"], out)
    print({k: v for k, v in out.items() if k != "details"})


if __name__ == "__main__":
    main()
