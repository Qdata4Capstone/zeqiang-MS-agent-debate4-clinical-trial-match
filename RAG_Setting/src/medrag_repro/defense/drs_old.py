from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from medrag_repro.retriever.contriever import ContrieverEncoder


def compute_drs_reference(clean_doc_texts: Sequence[str], encoder: ContrieverEncoder, M: int, eps: float = 1e-8) -> Dict[str, object]:
    X = encoder.encode(list(clean_doc_texts), normalize=False).astype(np.float64)
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std[std < eps] = 1.0
    Xs = (X - mean) / std
    cov = np.cov(Xs, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    M_eff = min(M, eigvals.shape[0])

    def score_vectors(texts: Sequence[str]) -> np.ndarray:
        Z = encoder.encode(list(texts), normalize=False).astype(np.float64)
        Zs = (Z - mean) / std
        scores = np.zeros(Zs.shape[0], dtype=np.float64)
        for i in range(M_eff):
            lam = float(eigvals[i])
            lam = lam if lam > eps else eps
            v = eigvecs[:, i]
            scores += np.abs(Zs @ v) / lam
        return scores

    clean_scores = score_vectors(clean_doc_texts)
    return {
        "mean": mean,
        "std": std,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "M": M_eff,
        "clean_scores": clean_scores,
        "score_fn": score_vectors,
    }
