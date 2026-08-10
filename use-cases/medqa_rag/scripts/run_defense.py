#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from medrag_repro.config import load_config
from medrag_repro.datamodels import CorpusDoc, PoisonDoc, QAItem
from medrag_repro.defense.drs import DRSDetector
from medrag_repro.defense.l2_distance import L2DistanceDetector
from medrag_repro.defense.l2_norm import L2NormDetector
from medrag_repro.defense.perplexity import PerplexityDetector
from medrag_repro.evaluation.rag_eval import evaluate_attack
from medrag_repro.llm.client import load_openai_client
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import load_index, retrieve_topk
from medrag_repro.utils.io import read_jsonl, write_json, write_jsonl
from medrag_repro.utils.seed import set_seed

ALL_METHODS = ["drs", "l2_norm", "l2_distance", "perplexity"]


def build_detector(method: str, cfg: dict, encoder: ContrieverEncoder):
    if method == "drs":
        return DRSDetector(
            encoder=encoder,
            M=cfg["drs"]["M"],
            clean_quantile=cfg["drs"]["clean_threshold_quantile"],
        )

    if method == "l2_norm":
        return L2NormDetector(
            encoder=encoder,
            lower_quantile=0.01,
            upper_quantile=0.99,
        )

    if method == "l2_distance":
        return L2DistanceDetector(
            encoder=encoder,
            clean_quantile=0.99,
        )

    if method == "perplexity":
        ppl_model = cfg.get("baseline", {}).get("perplexity_model", "distilgpt2")
        ppl_device = cfg.get("baseline", {}).get("perplexity_device", "cpu")
        return PerplexityDetector(
            model_name=ppl_model,
            device=ppl_device,
            lower_quantile=0.01,
            upper_quantile=0.99,
        )

    raise ValueError(f"Unknown defense method: {method}")


def run_one_defense(
    method: str,
    cfg: dict,
    client,
    encoder: ContrieverEncoder,
    index: dict,
    doc_ids: list,
    vectors: np.ndarray,
    doc_lookup: dict,
    clean_ref_texts: list,
    targets: list,
    poison: list,
) -> dict:
    """Fit `method`'s detector on the clean reference set, filter the poison
    docs it flags, and re-evaluate the attack with the survivors. Returns the
    full metrics dict (including per-poison-doc `details`) but does not
    write any files -- callers decide what to persist."""
    detector = build_detector(method, cfg, encoder)
    detector.fit(clean_ref_texts)

    poison_texts = [p.full_text for p in poison]
    poison_scores = detector.score_texts(poison_texts)
    detected = detector.detect(poison_texts)

    clean_detected = detector.detect(clean_ref_texts)
    clean_fpr = float(np.mean(clean_detected)) if len(clean_detected) else 0.0

    kept_poison = [p for p, d in zip(poison, detected) if not d]

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
        {
            "poison_id": p.poison_id,
            "target_qid": p.target_qid,
            "score": float(s),
            "detected": bool(d),
        }
        for p, s, d in zip(poison, poison_scores.tolist(), detected)
    ]

    return {
        "method": method,
        "n_clean_reference_docs": len(clean_ref_texts),
        "clean_fpr": clean_fpr,
        "poison_detection_rate": float(np.mean(detected)) if detected else 0.0,
        "n_poison_before": len(poison),
        "n_poison_after": len(kept_poison),
        "post_defense_attack_success_rate": post["attack_success_rate"],
        "post_defense_retrieval_precision": post["retrieval_precision"],
        "post_defense_retrieval_recall": post["retrieval_recall"],
        "post_defense_retrieval_f1": post["retrieval_f1"],
        "details": details,
        "kept_poison": kept_poison,
    }


def run_no_defense_baseline(
    cfg: dict,
    client,
    encoder: ContrieverEncoder,
    index: dict,
    doc_ids: list,
    vectors: np.ndarray,
    doc_lookup: dict,
    targets: list,
    poison: list,
) -> dict:
    """Attack metrics with every poison doc left in place (no detector
    applied) -- the "no defense" row a fair comparison table needs."""
    post = evaluate_attack(
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
    return {
        "method": "none",
        "n_poison_before": len(poison),
        "n_poison_after": len(poison),
        "post_defense_attack_success_rate": post["attack_success_rate"],
        "post_defense_retrieval_precision": post["retrieval_precision"],
        "post_defense_retrieval_recall": post["retrieval_recall"],
        "post_defense_retrieval_f1": post["retrieval_f1"],
    }


def format_comparison_table(results: list) -> str:
    columns = [
        ("Method", "method"),
        ("Detect rate", "poison_detection_rate"),
        ("Clean FPR", "clean_fpr"),
        ("Attack success", "post_defense_attack_success_rate"),
        ("Retrieval F1", "post_defense_retrieval_f1"),
    ]

    def cell(result: dict, key: str) -> str:
        value = result.get(key)
        if value is None:
            return "-"
        if key == "method":
            return str(value)
        return f"{value:.4f}"

    col_width = 14
    header = "  ".join(f"{name:<{col_width}}" for name, _ in columns)
    lines = [header, "-" * len(header)]
    for result in results:
        lines.append("  ".join(f"{cell(result, key):<{col_width}}" for _, key in columns))
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=[*ALL_METHODS, "all"])
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

    # 1) build clean reference docs (shared across every method, so every
    # detector -- and the "all" mode's whole comparison table -- is fit on
    # the exact same reference set)
    clean_ref_map = {}
    for qa in clean_queries:
        retrieved = retrieve_topk(
            qa.question,
            encoder,
            index,
            doc_ids,
            doc_lookup,
            cfg["retriever"]["top_k"],
        )
        for doc_id, _, text in retrieved:
            clean_ref_map.setdefault(doc_id, text)

    clean_ref_texts = list(clean_ref_map.values())

    artifact_dir = Path(cfg["paths"]["artifact_dir"])

    if args.method != "all":
        result = run_one_defense(
            args.method, cfg, client, encoder, index, doc_ids, vectors,
            doc_lookup, clean_ref_texts, targets, poison,
        )
        write_json(artifact_dir / f"{args.method}_metrics.json", {k: v for k, v in result.items() if k != "kept_poison"})
        write_jsonl(artifact_dir / f"{args.method}_kept_poison.jsonl", result["kept_poison"])
        print({k: v for k, v in result.items() if k not in ("details", "kept_poison")})
        return

    baseline = run_no_defense_baseline(
        cfg, client, encoder, index, doc_ids, vectors, doc_lookup, targets, poison,
    )
    results = [baseline]
    for method in ALL_METHODS:
        result = run_one_defense(
            method, cfg, client, encoder, index, doc_ids, vectors,
            doc_lookup, clean_ref_texts, targets, poison,
        )
        write_json(artifact_dir / f"{method}_metrics.json", {k: v for k, v in result.items() if k != "kept_poison"})
        write_jsonl(artifact_dir / f"{method}_kept_poison.jsonl", result["kept_poison"])
        results.append({k: v for k, v in result.items() if k not in ("details", "kept_poison")})

    write_json(artifact_dir / "all_defenses_metrics.json", results)

    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
