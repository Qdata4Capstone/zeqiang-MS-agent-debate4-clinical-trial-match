from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from medrag_repro.retriever.contriever import ContrieverEncoder


def build_index(vectors: np.ndarray, backend: str = "numpy") -> dict:
    if vectors.dtype != np.float32:
        vectors = vectors.astype("float32")
    return {
        "backend": backend,
        "vectors": vectors,
    }


def save_index(index_dir: str | Path, index: dict, doc_ids: Sequence[str]) -> None:
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    np.save(index_dir / "vectors.npy", index["vectors"])

    with (index_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"backend": index["backend"]}, f)

    with (index_dir / "doc_ids.json").open("w", encoding="utf-8") as f:
        json.dump(list(doc_ids), f)


def load_index(index_dir: str | Path) -> tuple[dict, list[str], np.ndarray]:
    index_dir = Path(index_dir)

    with (index_dir / "meta.json").open("r", encoding="utf-8") as f:
        meta = json.load(f)

    with (index_dir / "doc_ids.json").open("r", encoding="utf-8") as f:
        doc_ids = json.load(f)

    vectors = np.load(index_dir / "vectors.npy")

    return {"backend": meta.get("backend", "numpy"), "vectors": vectors}, doc_ids, vectors


def retrieve_topk(
    query: str,
    encoder: ContrieverEncoder,
    index: dict,
    doc_ids: Sequence[str],
    doc_lookup: Dict[str, str],
    k: int,
) -> List[Tuple[str, float, str]]:
    qv = encoder.encode([query])[0]   # [d]
    vecs = index["vectors"]           # [N, d]

    scores = vecs @ qv

    if k >= len(scores):
        top_idx = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, kth=k - 1)[:k]
        top_idx = part[np.argsort(-scores[part])]

    out: List[Tuple[str, float, str]] = []
    for i in top_idx.tolist():
        doc_id = doc_ids[i]
        out.append((doc_id, float(scores[i]), doc_lookup[doc_id]))

    return out