import argparse
import hashlib
import json
import os
import random
from copy import deepcopy

import torch

from poisonrag_experiment.drs import drs_score, drs_threshold
from poisonrag_experiment.ollama_utils import generate_json
from poisonrag_experiment.retrieval_utils import (
    MedCPTQueryEncoder,
    build_bm25_index,
    build_medcpt_corpus_index,
    dump_json,
    get_conditions,
    get_device,
    load_jsonl,
    load_qrels,
    load_queries_and_keywords,
    rank_query,
    recall_at_k,
)
from rag_attacks.poisonedrag_trial import (
    build_poison_text,
    choose_example_trial,
    corpus_entry_to_example,
    generate_poison_trials,
    select_target_patients,
)
from rag_defenses.defense_baselines import fit_two_sided_quantile, fit_upper_quantile
from rag_defenses.l2_distance import l2_distance_scores, leave_one_out_l2_distance_scores
from rag_defenses.l2_norm import l2_norm_scores
from rag_defenses.perplexity import PerplexityScorer


def parse_args():
    parser = argparse.ArgumentParser(description="Run PoisonRAG retrieval experiments on TrialGPT data.")
    parser.add_argument("--corpus", default="sigir", choices=["sigir", "trec_2021", "trec_2022"])
    parser.add_argument("--dataset_dir", default="dataset")
    parser.add_argument("--query_type", default="gpt-4-turbo")
    parser.add_argument("--num_targets", type=int, default=20)
    parser.add_argument("--poisons_per_patient", type=int, default=3)
    parser.add_argument("--retrieval_top_n", type=int, default=200)
    parser.add_argument("--fusion_k", type=int, default=20)
    parser.add_argument("--bm25_weight", type=float, default=1.0)
    parser.add_argument("--dense_weight", type=float, default=1.0)
    parser.add_argument("--target_seed", type=int, default=13)
    parser.add_argument("--ollama_model", default="qwen-2.5:7b-instruct")
    parser.add_argument("--ollama_base_url", default="http://localhost:11434")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--output_dir", default="results/poisonrag_experiment")
    parser.add_argument("--drs_ref_k", type=int, default=20)
    parser.add_argument("--drs_num_directions", type=int, default=16)
    parser.add_argument("--drs_quantile", type=float, default=0.99)
    parser.add_argument("--drs_power", type=float, default=1.0)
    parser.add_argument(
        "--drs_pool_reference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit one DRS model on clean reference documents pooled across all target queries "
        "(the paper's actual Algorithm 2), instead of a separate model per target query using "
        "only that query's own top --drs_ref_k docs. Pooling gives DRS up to "
        "--drs_ref_k * --num_targets reference documents instead of --drs_ref_k alone -- "
        "confirmed to catch more injected poison at no recall cost (see "
        "docs/drs-dual-pca-analysis.md), so it's the default. Pass --no-drs_pool_reference "
        "for the original per-query behavior.",
    )
    parser.add_argument(
        "--compare_defenses",
        action="store_true",
        help="Also evaluate L2-norm, L2-distance, and perplexity baseline defenses alongside DRS, "
        "using the same clean reference set (--drs_ref_k) and quantile (--drs_quantile) for a fair "
        "comparison, and print a comparison table.",
    )
    parser.add_argument(
        "--baseline_perplexity_model",
        default="distilgpt2",
        help="Causal LM used by the perplexity baseline (only loaded when --compare_defenses is set).",
    )
    return parser.parse_args()


def corpus_content_hash(corpus_entries):
    digest = hashlib.md5()
    for entry in corpus_entries:
        digest.update(entry["_id"].encode("utf-8"))
        digest.update(entry["title"].encode("utf-8"))
        digest.update(entry["text"].encode("utf-8"))
    return digest.hexdigest()[:12]


def get_paths(args):
    dataset_dir = os.path.join(args.dataset_dir, args.corpus)
    return {
        "dataset_dir": dataset_dir,
        "queries_path": os.path.join(dataset_dir, "queries.jsonl"),
        "corpus_path": os.path.join(dataset_dir, "corpus.jsonl"),
    }


def evaluate_rankings(rankings, qrels, ks=(50, 100, 200)):
    metrics = {f"recall@{k}": [] for k in ks}
    for qid, ranked in rankings.items():
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        positives = qrels[qid]
        for k in ks:
            metrics[f"recall@{k}"].append(recall_at_k(ranked_doc_ids, positives, k))

    return {
        metric: sum(values) / len(values) if values else 0.0
        for metric, values in metrics.items()
    }


def run_retrieval_for_queries(
    target_qids,
    id2queries,
    query_type,
    bm25,
    bm25_doc_ids,
    medcpt_index,
    query_encoder,
    top_n,
    fusion_k,
    bm25_weight,
    dense_weight,
):
    rankings = {}
    for qid in target_qids:
        conditions = get_conditions(id2queries, qid, query_type)
        rankings[qid] = rank_query(
            conditions=conditions,
            bm25=bm25,
            bm25_doc_ids=bm25_doc_ids,
            medcpt_index=medcpt_index,
            query_encoder=query_encoder,
            top_n=top_n,
            fusion_k=fusion_k,
            bm25_weight=bm25_weight,
            dense_weight=dense_weight,
        )
    return rankings


def apply_drs_defense(
    target_qids,
    clean_rankings,
    poisoned_rankings,
    medcpt_index_clean,
    medcpt_index_poisoned,
    ref_k,
    quantile,
    num_directions,
    power,
):
    defended_rankings = {}
    drs_metadata = {}

    for qid in target_qids:
        clean_ref_doc_ids = [doc_id for doc_id, _ in clean_rankings[qid][:ref_k]]
        clean_embeddings = [medcpt_index_clean.get_embedding(doc_id) for doc_id in clean_ref_doc_ids]
        model, clean_scores, threshold = drs_threshold(
            clean_embeddings=clean_embeddings,
            quantile=quantile,
            num_directions=num_directions,
            power=power,
        )

        filtered = []
        flagged_doc_ids = []
        for doc_id, score in poisoned_rankings[qid]:
            embedding = medcpt_index_poisoned.get_embedding(doc_id)
            score_drs = drs_score(embedding, model)
            if score_drs > threshold:
                flagged_doc_ids.append({"doc_id": doc_id, "drs_score": score_drs})
                continue
            filtered.append((doc_id, score))

        defended_rankings[qid] = filtered
        drs_metadata[qid] = {
            "threshold": threshold,
            "clean_scores": clean_scores,
            "flagged": flagged_doc_ids,
        }

    return defended_rankings, drs_metadata


def apply_drs_defense_pooled(
    target_qids,
    clean_rankings,
    poisoned_rankings,
    medcpt_index_clean,
    medcpt_index_poisoned,
    ref_k,
    quantile,
    num_directions,
    power,
):
    """Paper-faithful Algorithm 2: retrieve top-`ref_k` clean documents for
    *every* protected query, pool them into one combined, deduplicated
    reference set, and fit a single shared DRS model applied to all target
    queries -- as opposed to apply_drs_defense's per-query variant (a
    separate model per target, using only that target's own top-`ref_k`
    docs), which structurally caps the achievable reference-set size at
    `ref_k` alone instead of up to `ref_k * len(target_qids)`."""
    pooled_doc_ids = []
    seen = set()
    for qid in target_qids:
        for doc_id, _ in clean_rankings[qid][:ref_k]:
            if doc_id not in seen:
                seen.add(doc_id)
                pooled_doc_ids.append(doc_id)

    clean_embeddings = [medcpt_index_clean.get_embedding(doc_id) for doc_id in pooled_doc_ids]
    model, clean_scores, threshold = drs_threshold(
        clean_embeddings=clean_embeddings,
        quantile=quantile,
        num_directions=num_directions,
        power=power,
    )

    defended_rankings = {}
    flagged_by_qid = {}
    for qid in target_qids:
        filtered = []
        flagged_doc_ids = []
        for doc_id, score in poisoned_rankings[qid]:
            embedding = medcpt_index_poisoned.get_embedding(doc_id)
            score_drs = drs_score(embedding, model)
            if score_drs > threshold:
                flagged_doc_ids.append({"doc_id": doc_id, "drs_score": score_drs})
                continue
            filtered.append((doc_id, score))
        defended_rankings[qid] = filtered
        flagged_by_qid[qid] = flagged_doc_ids

    drs_metadata = {
        "pooled_reference_size": len(pooled_doc_ids),
        "num_directions": model["eigenvectors"].shape[1],
        "threshold": threshold,
        "clean_scores": clean_scores,
        "flagged_by_qid": flagged_by_qid,
    }
    return defended_rankings, drs_metadata


def apply_l2_norm_defense(
    target_qids,
    clean_rankings,
    poisoned_rankings,
    medcpt_index_clean,
    medcpt_index_poisoned,
    ref_k,
    quantile,
):
    """Baseline: flag candidates whose embedding L2-norm falls outside the
    two-sided quantile range of the clean reference set's norms (mirrors
    rag_defenses.l2_norm.L2NormDetector, applied directly to the MedCPT
    embeddings already computed for retrieval, instead of re-encoding text)."""
    defended_rankings = {}
    metadata = {}

    for qid in target_qids:
        clean_ref_doc_ids = [doc_id for doc_id, _ in clean_rankings[qid][:ref_k]]
        clean_embeddings = torch.stack(
            [torch.from_numpy(medcpt_index_clean.get_embedding(doc_id)) for doc_id in clean_ref_doc_ids]
        )
        clean_scores = l2_norm_scores(clean_embeddings)
        stats = fit_two_sided_quantile(clean_scores, quantile=quantile)

        filtered = []
        flagged_doc_ids = []
        for doc_id, score in poisoned_rankings[qid]:
            embedding = torch.from_numpy(medcpt_index_poisoned.get_embedding(doc_id))
            candidate_score = l2_norm_scores(embedding).item()
            if candidate_score < stats.lower_threshold or candidate_score > stats.upper_threshold:
                flagged_doc_ids.append({"doc_id": doc_id, "l2_norm_score": candidate_score})
                continue
            filtered.append((doc_id, score))

        defended_rankings[qid] = filtered
        metadata[qid] = {
            "lower_threshold": stats.lower_threshold,
            "upper_threshold": stats.upper_threshold,
            "flagged": flagged_doc_ids,
        }

    return defended_rankings, metadata


def apply_l2_distance_defense(
    target_qids,
    clean_rankings,
    poisoned_rankings,
    medcpt_index_clean,
    medcpt_index_poisoned,
    ref_k,
    quantile,
):
    """Baseline: flag candidates whose nearest-neighbor distance to the clean
    reference set exceeds the upper quantile of the reference set's own
    leave-one-out nearest-neighbor distances (mirrors
    rag_defenses.l2_distance's strategyqa_agent-side functions)."""
    defended_rankings = {}
    metadata = {}

    for qid in target_qids:
        clean_ref_doc_ids = [doc_id for doc_id, _ in clean_rankings[qid][:ref_k]]
        clean_embeddings = torch.stack(
            [torch.from_numpy(medcpt_index_clean.get_embedding(doc_id)) for doc_id in clean_ref_doc_ids]
        )
        clean_scores = leave_one_out_l2_distance_scores(clean_embeddings)
        stats = fit_upper_quantile(clean_scores, quantile=quantile)

        filtered = []
        flagged_doc_ids = []
        for doc_id, score in poisoned_rankings[qid]:
            embedding = torch.from_numpy(medcpt_index_poisoned.get_embedding(doc_id))
            candidate_score = l2_distance_scores(embedding, clean_embeddings).item()
            if candidate_score > stats.threshold:
                flagged_doc_ids.append({"doc_id": doc_id, "l2_distance_score": candidate_score})
                continue
            filtered.append((doc_id, score))

        defended_rankings[qid] = filtered
        metadata[qid] = {"threshold": stats.threshold, "flagged": flagged_doc_ids}

    return defended_rankings, metadata


def _doc_text(entry):
    return f"{entry['title']} {entry['text']}"


def apply_perplexity_defense(
    target_qids,
    clean_rankings,
    poisoned_rankings,
    corpus_by_id,
    poisoned_corpus_by_id,
    ref_k,
    quantile,
    model_name,
):
    """Baseline: flag candidates whose causal-LM perplexity falls outside the
    two-sided quantile range of the clean reference set's perplexities
    (mirrors rag_defenses.perplexity.PerplexityScorer). Operates on raw
    text, not embeddings, so it loads its own LM."""
    scorer = PerplexityScorer(model_name=model_name, device=get_device())
    defended_rankings = {}
    metadata = {}

    for qid in target_qids:
        clean_ref_doc_ids = [doc_id for doc_id, _ in clean_rankings[qid][:ref_k]]
        clean_texts = [_doc_text(corpus_by_id[doc_id]) for doc_id in clean_ref_doc_ids]
        clean_scores = scorer.score_texts(clean_texts)
        stats = fit_two_sided_quantile(clean_scores, quantile=quantile)

        filtered = []
        flagged_doc_ids = []
        for doc_id, score in poisoned_rankings[qid]:
            text = _doc_text(poisoned_corpus_by_id[doc_id])
            candidate_score = scorer.score_texts([text]).item()
            if candidate_score < stats.lower_threshold or candidate_score > stats.upper_threshold:
                flagged_doc_ids.append({"doc_id": doc_id, "perplexity": candidate_score})
                continue
            filtered.append((doc_id, score))

        defended_rankings[qid] = filtered
        metadata[qid] = {
            "lower_threshold": stats.lower_threshold,
            "upper_threshold": stats.upper_threshold,
            "flagged": flagged_doc_ids,
        }

    return defended_rankings, metadata


def format_comparison_table(metrics, ks=(50, 100, 200)):
    rows = [
        ("clean (no attack)", "clean_baseline"),
        ("poisoned, no defense", "poisoned"),
        ("poisoned + DRS", "poisoned_with_drs"),
        ("poisoned + L2-norm", "poisoned_with_l2_norm"),
        ("poisoned + L2-distance", "poisoned_with_l2_distance"),
        ("poisoned + perplexity", "poisoned_with_perplexity"),
    ]
    rows = [(label, key) for label, key in rows if key in metrics]

    label_width = max(len(label) for label, _ in rows)
    col_width = 11
    header = f"{'Method':<{label_width}}  " + "  ".join(f"{f'recall@{k}':<{col_width}}" for k in ks)
    lines = [header, "-" * len(header)]
    for label, key in rows:
        values = "  ".join(f"{metrics[key][f'recall@{k}']:<{col_width}.4f}" for k in ks)
        lines.append(f"{label:<{label_width}}  {values}")
    return "\n".join(lines)


def collect_attack_stats(target_qids, poisoned_rankings):
    stats = {}
    for qid in target_qids:
        poison_positions = []
        for idx, (doc_id, score) in enumerate(poisoned_rankings[qid], start=1):
            if doc_id.startswith(f"POISON-{qid}-"):
                poison_positions.append({"doc_id": doc_id, "rank": idx, "score": score})
        stats[qid] = poison_positions
    return stats


def main():
    args = parse_args()
    paths = get_paths(args)
    os.makedirs(args.output_dir, exist_ok=True)

    queries, id2queries = load_queries_and_keywords(paths["dataset_dir"])
    qrels = load_qrels(paths["dataset_dir"])
    clean_corpus_entries = load_jsonl(paths["corpus_path"])
    corpus_by_id = {entry["_id"]: entry for entry in clean_corpus_entries}

    target_qids = select_target_patients(
        query_ids=list(queries.keys()),
        qrels=qrels,
        num_targets=args.num_targets,
        seed=args.target_seed,
    )
    dump_json(os.path.join(args.output_dir, "target_patients.json"), target_qids)

    poison_trials = generate_poison_trials(
        args=args,
        target_qids=target_qids,
        queries=queries,
        id2queries=id2queries,
        qrels=qrels,
        corpus_by_id=corpus_by_id,
    )
    dump_json(os.path.join(args.output_dir, "poison_trials.json"), poison_trials)

    poisoned_corpus_entries = deepcopy(clean_corpus_entries) + poison_trials
    poisoned_corpus_path = os.path.join(args.output_dir, f"{args.corpus}_poisoned_corpus.jsonl")
    with open(poisoned_corpus_path, "w") as handle:
        for entry in poisoned_corpus_entries:
            handle.write(json.dumps(entry) + "\n")

    clean_bm25, clean_bm25_doc_ids = build_bm25_index(clean_corpus_entries)
    poisoned_bm25, poisoned_bm25_doc_ids = build_bm25_index(poisoned_corpus_entries)

    cache_dir = os.path.join(args.output_dir, "cache")
    clean_cache_key = corpus_content_hash(clean_corpus_entries)
    poisoned_cache_key = corpus_content_hash(poisoned_corpus_entries)

    clean_medcpt_index = build_medcpt_corpus_index(
        corpus_entries=clean_corpus_entries,
        cache_dir=cache_dir,
        cache_key=f"{args.corpus}_clean_{clean_cache_key}",
    )
    poisoned_medcpt_index = build_medcpt_corpus_index(
        corpus_entries=poisoned_corpus_entries,
        cache_dir=cache_dir,
        cache_key=f"{args.corpus}_poisoned_{poisoned_cache_key}",
    )

    query_encoder = MedCPTQueryEncoder()
    clean_rankings = run_retrieval_for_queries(
        target_qids=target_qids,
        id2queries=id2queries,
        query_type=args.query_type,
        bm25=clean_bm25,
        bm25_doc_ids=clean_bm25_doc_ids,
        medcpt_index=clean_medcpt_index,
        query_encoder=query_encoder,
        top_n=args.retrieval_top_n,
        fusion_k=args.fusion_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )
    poisoned_rankings = run_retrieval_for_queries(
        target_qids=target_qids,
        id2queries=id2queries,
        query_type=args.query_type,
        bm25=poisoned_bm25,
        bm25_doc_ids=poisoned_bm25_doc_ids,
        medcpt_index=poisoned_medcpt_index,
        query_encoder=query_encoder,
        top_n=args.retrieval_top_n,
        fusion_k=args.fusion_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
    )
    drs_defense_fn = apply_drs_defense_pooled if args.drs_pool_reference else apply_drs_defense
    defended_rankings, drs_metadata = drs_defense_fn(
        target_qids=target_qids,
        clean_rankings=clean_rankings,
        poisoned_rankings=poisoned_rankings,
        medcpt_index_clean=clean_medcpt_index,
        medcpt_index_poisoned=poisoned_medcpt_index,
        ref_k=args.drs_ref_k,
        quantile=args.drs_quantile,
        num_directions=args.drs_num_directions,
        power=args.drs_power,
    )

    metrics = {
        "clean_baseline": evaluate_rankings(clean_rankings, qrels),
        "poisoned": evaluate_rankings(poisoned_rankings, qrels),
        "poisoned_with_drs": evaluate_rankings(defended_rankings, qrels),
    }
    attack_stats = collect_attack_stats(target_qids, poisoned_rankings)

    dump_json(os.path.join(args.output_dir, "clean_rankings.json"), clean_rankings)
    dump_json(os.path.join(args.output_dir, "poisoned_rankings.json"), poisoned_rankings)
    dump_json(os.path.join(args.output_dir, "drs_rankings.json"), defended_rankings)
    dump_json(os.path.join(args.output_dir, "drs_metadata.json"), drs_metadata)
    dump_json(os.path.join(args.output_dir, "attack_stats.json"), attack_stats)

    if args.compare_defenses:
        poisoned_corpus_by_id = {entry["_id"]: entry for entry in poisoned_corpus_entries}

        l2_norm_rankings, l2_norm_metadata = apply_l2_norm_defense(
            target_qids=target_qids,
            clean_rankings=clean_rankings,
            poisoned_rankings=poisoned_rankings,
            medcpt_index_clean=clean_medcpt_index,
            medcpt_index_poisoned=poisoned_medcpt_index,
            ref_k=args.drs_ref_k,
            quantile=args.drs_quantile,
        )
        l2_distance_rankings, l2_distance_metadata = apply_l2_distance_defense(
            target_qids=target_qids,
            clean_rankings=clean_rankings,
            poisoned_rankings=poisoned_rankings,
            medcpt_index_clean=clean_medcpt_index,
            medcpt_index_poisoned=poisoned_medcpt_index,
            ref_k=args.drs_ref_k,
            quantile=args.drs_quantile,
        )
        perplexity_rankings, perplexity_metadata = apply_perplexity_defense(
            target_qids=target_qids,
            clean_rankings=clean_rankings,
            poisoned_rankings=poisoned_rankings,
            corpus_by_id=corpus_by_id,
            poisoned_corpus_by_id=poisoned_corpus_by_id,
            ref_k=args.drs_ref_k,
            quantile=args.drs_quantile,
            model_name=args.baseline_perplexity_model,
        )

        metrics["poisoned_with_l2_norm"] = evaluate_rankings(l2_norm_rankings, qrels)
        metrics["poisoned_with_l2_distance"] = evaluate_rankings(l2_distance_rankings, qrels)
        metrics["poisoned_with_perplexity"] = evaluate_rankings(perplexity_rankings, qrels)

        dump_json(os.path.join(args.output_dir, "l2_norm_rankings.json"), l2_norm_rankings)
        dump_json(os.path.join(args.output_dir, "l2_norm_metadata.json"), l2_norm_metadata)
        dump_json(os.path.join(args.output_dir, "l2_distance_rankings.json"), l2_distance_rankings)
        dump_json(os.path.join(args.output_dir, "l2_distance_metadata.json"), l2_distance_metadata)
        dump_json(os.path.join(args.output_dir, "perplexity_rankings.json"), perplexity_rankings)
        dump_json(os.path.join(args.output_dir, "perplexity_metadata.json"), perplexity_metadata)

    dump_json(os.path.join(args.output_dir, "metrics.json"), metrics)

    print(json.dumps(metrics, indent=2))
    if args.compare_defenses:
        print()
        print(format_comparison_table(metrics))


if __name__ == "__main__":
    main()
