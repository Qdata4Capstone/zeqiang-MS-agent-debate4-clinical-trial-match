import argparse
import hashlib
import json
import os
import random
from copy import deepcopy

from poisonrag_experiment.drs import drs_score, drs_threshold
from poisonrag_experiment.ollama_utils import generate_json
from poisonrag_experiment.retrieval_utils import (
    MedCPTQueryEncoder,
    build_bm25_index,
    build_medcpt_corpus_index,
    dump_json,
    get_conditions,
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
    defended_rankings, drs_metadata = apply_drs_defense(
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
    dump_json(os.path.join(args.output_dir, "metrics.json"), metrics)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
