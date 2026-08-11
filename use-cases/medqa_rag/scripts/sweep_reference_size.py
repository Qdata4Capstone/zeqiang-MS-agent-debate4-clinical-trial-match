#!/usr/bin/env python3
"""Sweep DRS's clean-reference-set size (and M) against the baseline
defenses to see where DRS starts beating them.

`docs/drs-dual-pca-analysis.md` found DRS loses to L2-norm/L2-distance at
this repo's demo scale (n=29 clean reference docs, d=768) but ties/wins at
larger scale (trial_retrieval, n=521). The paper's own setup pools ~1,000
clean reference docs (M=100, 300 clean queries, k=5). This script reuses
the shared data/index/poison prep from `configs/sweep.yaml` (run
prepare_data.py / build_index.py / generate_poison.py first) and sweeps the
number of clean queries pooled into the reference set -- and, for DRS, M --
to trace out that crossover, without paying for the full LLM-based attack
eval at every point (pass --with_attack_eval to also get that, slow).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_defense import build_detector  # noqa: E402

from medrag_repro.config import load_config
from medrag_repro.datamodels import CorpusDoc, PoisonDoc, QAItem
from medrag_repro.evaluation.rag_eval import evaluate_attack
from medrag_repro.llm.client import load_openai_client
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import load_index, retrieve_topk
from medrag_repro.utils.io import read_jsonl, write_json
from medrag_repro.utils.seed import set_seed

BASELINE_METHODS = ["l2_norm", "l2_distance", "perplexity"]


class CachingEncoder:
    """Wraps a ContrieverEncoder and memoizes embeddings by text.

    Reference-set subsets at growing query counts overlap heavily (they're
    prefixes of the same pooled doc set), and DRS/L2-norm/L2-distance all
    re-encode their clean texts from scratch on every `fit()`. Memoizing
    avoids re-running Contriever over the same docs at every sweep point.
    """

    def __init__(self, encoder: ContrieverEncoder):
        self.encoder = encoder
        self._cache: dict[str, np.ndarray] = {}

    def encode(self, texts, normalize: bool = False) -> np.ndarray:
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            vecs = self.encoder.encode(uncached, normalize=normalize)
            for t, v in zip(uncached, vecs):
                self._cache[t] = v
        return np.stack([self._cache[t] for t in texts])


def fit_perplexity_cached(detector, texts: list[str], score_cache: dict[str, float]) -> None:
    """Same idea as CachingEncoder but for PerplexityDetector, which scores
    texts directly (no embedding step to intercept)."""
    uncached = [t for t in texts if t not in score_cache]
    if uncached:
        new_scores = detector.score_texts(uncached)
        for t, s in zip(uncached, new_scores.tolist()):
            score_cache[t] = s
    scores = np.array([score_cache[t] for t in texts], dtype=np.float64)
    detector.clean_scores = scores
    detector.fit_thresholds_from_scores(scores)


def build_pooled_reference(clean_queries, encoder, index, doc_ids, doc_lookup, top_k):
    """Retrieve top-k docs per clean query, in query order, pooling into a
    dict of unique doc_id -> text in first-seen order. Returns the pooled
    dict plus, for each query index, the pooled size *after* that query --
    so "first N queries" always maps to a stable prefix of pooled docs."""
    pooled: dict[str, str] = {}
    pooled_size_after_query = []
    for qa in clean_queries:
        retrieved = retrieve_topk(qa.question, encoder, index, doc_ids, doc_lookup, top_k)
        for doc_id, _, text in retrieved:
            pooled.setdefault(doc_id, text)
        pooled_size_after_query.append(len(pooled))
    return pooled, pooled_size_after_query


def evaluate_point(
    method: str,
    ref_texts: list[str],
    cfg: dict,
    encoder,
    poison,
    M: int | None,
    perplexity_detector,
    perplexity_cache: dict[str, float],
) -> dict:
    if method == "drs":
        detector = build_detector("drs", {**cfg, "drs": {**cfg["drs"], "M": M}}, encoder)
        detector.fit(ref_texts)
    elif method == "perplexity":
        detector = perplexity_detector
        fit_perplexity_cached(detector, ref_texts, perplexity_cache)
    else:
        detector = build_detector(method, cfg, encoder)
        detector.fit(ref_texts)

    poison_texts = [p.full_text for p in poison]
    detected = detector.detect(poison_texts)
    clean_detected = detector.detect(ref_texts)

    return {
        "method": method,
        "M": M if method == "drs" else None,
        "n_ref_docs": len(ref_texts),
        "poison_detection_rate": float(np.mean(detected)) if detected else 0.0,
        "clean_fpr": float(np.mean(clean_detected)) if clean_detected else 0.0,
        "n_poison_caught": int(np.sum(detected)),
        "n_poison_total": len(poison),
    }


def maybe_add_attack_eval(
    result: dict,
    detector_detected_mask: list[bool],
    poison,
    cfg,
    client,
    encoder,
    index,
    doc_ids,
    vectors,
    doc_lookup,
    targets,
) -> None:
    kept_poison = [p for p, d in zip(poison, detector_detected_mask) if not d]
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
    result["post_defense_attack_success_rate"] = post["attack_success_rate"]
    result["post_defense_retrieval_f1"] = post["retrieval_f1"]


def format_table(rows: list[dict]) -> str:
    columns = [
        ("RefQ", "n_ref_queries"),
        ("RefDocs", "n_ref_docs"),
        ("Method", "method"),
        ("M", "M"),
        ("Detect", "poison_detection_rate"),
        ("CleanFPR", "clean_fpr"),
    ]
    col_width = 10

    def cell(row: dict, key: str) -> str:
        value = row.get(key)
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "  ".join(f"{name:<{col_width}}" for name, _ in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append("  ".join(f"{cell(row, key):<{col_width}}" for _, key in columns))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweep.yaml")
    ap.add_argument(
        "--ref_sizes",
        default="29,50,100,200,300",
        help="comma-separated numbers of clean queries to pool into the reference set",
    )
    ap.add_argument(
        "--m_values",
        default="10,50,100",
        help="comma-separated DRS M values (number of low-variance directions) to try at each ref size",
    )
    ap.add_argument(
        "--with_attack_eval",
        action="store_true",
        help="also run the full LLM-based attack eval at every sweep point (slow -- needs Ollama running)",
    )
    ap.add_argument("--output", default=None, help="defaults to <artifact_dir>/reference_size_sweep.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    ref_sizes = sorted({int(x) for x in args.ref_sizes.split(",") if x.strip()})
    m_values = sorted({int(x) for x in args.m_values.split(",") if x.strip()})

    clean_queries = [QAItem(**row) for row in read_jsonl(cfg["paths"]["clean_queries"])]
    targets = [QAItem(**row) for row in read_jsonl(cfg["paths"]["targets"])]
    corpus = [CorpusDoc(**row) for row in read_jsonl(cfg["paths"]["pubmed_corpus"])]
    poison = [PoisonDoc(**row) for row in read_jsonl(cfg["paths"]["poison_docs"])]

    max_ref_size = max(ref_sizes)
    if max_ref_size > len(clean_queries):
        raise ValueError(
            f"--ref_sizes asks for {max_ref_size} clean queries but only "
            f"{len(clean_queries)} are available in {cfg['paths']['clean_queries']}"
        )

    doc_lookup = {d.doc_id: d.text for d in corpus}
    index, doc_ids, vectors = load_index(cfg["paths"]["index_dir"])

    raw_encoder = ContrieverEncoder(
        model_name=cfg["retriever"]["model_name"],
        batch_size=cfg["retriever"]["batch_size"],
        device=cfg["retriever"]["device"],
    )
    encoder = CachingEncoder(raw_encoder)

    client = load_openai_client() if args.with_attack_eval else None

    print(f"Pooling clean reference docs from up to {max_ref_size} queries...")
    pooled, pooled_size_after_query = build_pooled_reference(
        clean_queries[:max_ref_size], raw_encoder, index, doc_ids, doc_lookup, cfg["retriever"]["top_k"],
    )
    pooled_items = list(pooled.items())  # first-seen order, stable prefixes

    perplexity_cache: dict[str, float] = {}
    perplexity_detector = build_detector("perplexity", cfg, encoder)
    rows: list[dict] = []

    for ref_size in ref_sizes:
        n_pooled = pooled_size_after_query[ref_size - 1]
        ref_texts = [text for _, text in pooled_items[:n_pooled]]
        print(f"\n=== ref_size={ref_size} queries -> {len(ref_texts)} pooled clean docs ===")

        for method in BASELINE_METHODS:
            t0 = time.time()
            result = evaluate_point(method, ref_texts, cfg, encoder, poison, None, perplexity_detector, perplexity_cache)
            result["n_ref_queries"] = ref_size
            print(f"  {method:<12} detect={result['poison_detection_rate']:.4f} "
                  f"fpr={result['clean_fpr']:.4f} ({time.time() - t0:.1f}s)")
            rows.append(result)

        for M in m_values:
            clipped_M = min(M, len(ref_texts) - 1)
            if clipped_M < 1:
                print(f"  drs (M={M}) skipped: ref set too small ({len(ref_texts)} docs)")
                continue
            t0 = time.time()
            result = evaluate_point("drs", ref_texts, cfg, encoder, poison, clipped_M, perplexity_detector, perplexity_cache)
            result["n_ref_queries"] = ref_size
            if clipped_M != M:
                result["M_requested"] = M
            print(f"  drs (M={clipped_M:<4}) detect={result['poison_detection_rate']:.4f} "
                  f"fpr={result['clean_fpr']:.4f} ({time.time() - t0:.1f}s)")
            rows.append(result)

    if args.with_attack_eval:
        print("\nRunning full LLM-based attack eval at every sweep point (slow)...")
        for row in rows:
            ref_size = row["n_ref_queries"]
            n_pooled = pooled_size_after_query[ref_size - 1]
            ref_texts = [text for _, text in pooled_items[:n_pooled]]
            if row["method"] == "drs":
                detector = build_detector("drs", {**cfg, "drs": {**cfg["drs"], "M": row["M"]}}, encoder)
                detector.fit(ref_texts)
            elif row["method"] == "perplexity":
                detector = perplexity_detector
                fit_perplexity_cached(detector, ref_texts, perplexity_cache)
            else:
                detector = build_detector(row["method"], cfg, encoder)
                detector.fit(ref_texts)
            poison_texts = [p.full_text for p in poison]
            detected = detector.detect(poison_texts)
            maybe_add_attack_eval(
                row, detected, poison, cfg, client, encoder, index, doc_ids, vectors, doc_lookup, targets,
            )

    print("\n" + format_table(rows))

    output_path = Path(args.output) if args.output else Path(cfg["paths"]["artifact_dir"]) / "reference_size_sweep.json"
    write_json(output_path, rows)
    print(f"\nWrote {len(rows)} sweep points to {output_path}")


if __name__ == "__main__":
    main()
