#!/usr/bin/env python3
from __future__ import annotations

import argparse

from medrag_repro.config import load_config
from medrag_repro.datamodels import CorpusDoc
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import build_index, save_index
from medrag_repro.utils.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)

    docs = [CorpusDoc(**row) for row in read_jsonl(cfg["paths"]["pubmed_corpus"])]

    encoder = ContrieverEncoder(
        model_name=cfg["retriever"]["model_name"],
        batch_size=cfg["retriever"]["batch_size"],
        device=cfg["retriever"]["device"],
    )

    texts = [d.text for d in docs]
    vectors = encoder.encode(texts)

    index = build_index(
        vectors,
        backend=cfg["retriever"].get("backend", "numpy"),
    )

    save_index(
        cfg["paths"]["index_dir"],
        index,
        [d.doc_id for d in docs],
    )

    print(f"Indexed {len(docs)} docs with backend={cfg['retriever'].get('backend', 'numpy')}")


if __name__ == "__main__":
    main()