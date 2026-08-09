import json
import os

import faiss
import numpy as np
import torch
import tqdm
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from rag_infra.data.jsonl import dump_json, load_jsonl, load_qrels, load_queries_and_keywords


def get_conditions(id2queries, qid, query_type):
    if query_type in ["raw", "human_summary"]:
        return [id2queries[qid][query_type]]
    if "turbo" in query_type:
        return id2queries[qid][query_type]["conditions"]
    if query_type.startswith("Clinician"):
        return id2queries[qid].get(query_type, [])
    raise ValueError(f"Unsupported query type: {query_type}")


def _tokenized_entry(entry):
    metadata = entry.get("metadata", {})
    diseases_list = metadata.get("diseases_list", [])
    tokens = word_tokenize(entry["title"].lower()) * 3
    for disease in diseases_list:
        tokens += word_tokenize(disease.lower()) * 2
    tokens += word_tokenize(entry["text"].lower())
    return tokens


def build_bm25_index(corpus_entries):
    tokenized_corpus = []
    doc_ids = []
    for entry in corpus_entries:
        doc_ids.append(entry["_id"])
        tokenized_corpus.append(_tokenized_entry(entry))
    return BM25Okapi(tokenized_corpus), doc_ids


class MedCPTCorpusIndex:
    def __init__(self, embeddings, doc_ids):
        self.embeddings = embeddings.astype(np.float32)
        self.doc_ids = doc_ids
        self.doc_id_to_pos = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embeddings, k):
        return self.index.search(query_embeddings.astype(np.float32), k=k)

    def get_embedding(self, doc_id):
        return self.embeddings[self.doc_id_to_pos[doc_id]]


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_medcpt_corpus_index(
    corpus_entries,
    cache_dir,
    cache_key,
    model_name="ncbi/MedCPT-Article-Encoder",
):
    os.makedirs(cache_dir, exist_ok=True)
    embeddings_path = os.path.join(cache_dir, f"{cache_key}_embeddings.npy")
    doc_ids_path = os.path.join(cache_dir, f"{cache_key}_doc_ids.json")

    if os.path.exists(embeddings_path) and os.path.exists(doc_ids_path):
        embeddings = np.load(embeddings_path)
        doc_ids = json.load(open(doc_ids_path))
        return MedCPTCorpusIndex(embeddings, doc_ids)

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    doc_ids = []
    for entry in tqdm.tqdm(corpus_entries, desc="Encoding corpus with MedCPT"):
        doc_ids.append(entry["_id"])
        with torch.no_grad():
            encoded = tokenizer(
                [[entry["title"], entry["text"]]],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            embed = model(**encoded).last_hidden_state[:, 0, :]
            embeddings.append(embed[0].detach().cpu().numpy())

    embeddings = np.array(embeddings, dtype=np.float32)
    np.save(embeddings_path, embeddings)
    dump_json(doc_ids_path, doc_ids)
    return MedCPTCorpusIndex(embeddings, doc_ids)


class MedCPTQueryEncoder:
    def __init__(self, model_name="ncbi/MedCPT-Query-Encoder"):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, conditions):
        with torch.no_grad():
            encoded = self.tokenizer(
                conditions,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=256,
            ).to(self.device)
            return self.model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()


def reciprocal_rank_fusion(
    bm25_rankings,
    dense_rankings,
    fusion_k=20,
    bm25_weight=1.0,
    dense_weight=1.0,
):
    scores = {}
    for condition_idx, ranking in enumerate(bm25_rankings):
        for rank, doc_id in enumerate(ranking):
            scores.setdefault(doc_id, 0.0)
            scores[doc_id] += bm25_weight * (1.0 / (rank + fusion_k)) * (1.0 / (condition_idx + 1))

    for condition_idx, ranking in enumerate(dense_rankings):
        for rank, doc_id in enumerate(ranking):
            scores.setdefault(doc_id, 0.0)
            scores[doc_id] += dense_weight * (1.0 / (rank + fusion_k)) * (1.0 / (condition_idx + 1))

    return sorted(scores.items(), key=lambda item: -item[1])


def rank_query(
    conditions,
    bm25,
    bm25_doc_ids,
    medcpt_index,
    query_encoder,
    top_n=100,
    fusion_k=20,
    bm25_weight=1.0,
    dense_weight=1.0,
):
    if not conditions:
        return []

    bm25_rankings = []
    for condition in conditions:
        tokens = word_tokenize(condition.lower())
        bm25_rankings.append(bm25.get_top_n(tokens, bm25_doc_ids, n=top_n))

    query_embeddings = query_encoder.encode(conditions)
    _, dense_indices = medcpt_index.search(query_embeddings, k=top_n)
    dense_rankings = []
    for indices in dense_indices:
        dense_rankings.append([medcpt_index.doc_ids[idx] for idx in indices])

    fused = reciprocal_rank_fusion(
        bm25_rankings=bm25_rankings,
        dense_rankings=dense_rankings,
        fusion_k=fusion_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )
    return fused


def recall_at_k(ranked_doc_ids, positives, k):
    positive_ids = {doc_id for doc_id, score in positives.items() if score > 0}
    if not positive_ids:
        return 0.0
    hits = sum(1 for doc_id in ranked_doc_ids[:k] if doc_id in positive_ids)
    return hits / len(positive_ids)
