#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse

from tqdm import tqdm

from medrag_repro.config import load_config
from medrag_repro.data.medqa_loader import extract_medqa_records, sample_targets_and_clean_queries, try_load_medqa
from medrag_repro.data.pubmed_loader import extract_pubmed_doc, iter_pubmed_rows
from medrag_repro.utils.io import write_jsonl
from medrag_repro.utils.seed import set_seed
from medrag_repro.utils.text import sha256_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    # MedQA
    ds = try_load_medqa()
    items = extract_medqa_records(ds)
    targets, clean_queries = sample_targets_and_clean_queries(
        items,
        cfg["medqa"]["n_targets"],
        cfg["medqa"]["n_clean_queries"],
        cfg["seed"],
    )
    write_jsonl(cfg["paths"]["medqa_all"], items)
    write_jsonl(cfg["paths"]["targets"], targets)
    write_jsonl(cfg["paths"]["clean_queries"], clean_queries)

    # PubMed
    docs = []
    seen = set()
    max_docs = int(cfg["pubmed"]["max_docs"])
    min_abs_words = int(cfg["pubmed"]["min_abs_words"])
    for i, row in enumerate(tqdm(iter_pubmed_rows(cfg["pubmed"]["dataset_name"], streaming=True), desc="Streaming PubMed")):
        doc = extract_pubmed_doc(row, i)
        if doc is None:
            continue
        if len(doc.abstract.split()) < min_abs_words:
            continue
        h = sha256_text(doc.text)
        if h in seen:
            continue
        seen.add(h)
        docs.append(doc)
        if len(docs) >= max_docs:
            break
    write_jsonl(cfg["paths"]["pubmed_corpus"], docs)

    print(f"Saved {len(items)} MedQA items")
    print(f"Saved {len(targets)} targets")
    print(f"Saved {len(clean_queries)} clean queries")
    print(f"Saved {len(docs)} PubMed docs")


if __name__ == "__main__":
    main()
