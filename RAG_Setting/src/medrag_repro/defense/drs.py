from __future__ import annotations

from typing import Sequence

import numpy as np

from medrag_repro.defense.common import BaseDetector
from medrag_repro.retriever.contriever import ContrieverEncoder


class DRSDetector(BaseDetector):
    def __init__(self, encoder: ContrieverEncoder, M: int = 100, clean_quantile: float = 0.99, eps: float = 1e-8):
        super().__init__(two_sided=False, upper_quantile=clean_quantile)
        self.encoder = encoder
        self.M = M
        self.eps = eps
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.eigvals: np.ndarray | None = None
        self.eigvecs: np.ndarray | None = None
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        X = self.encoder.encode(list(clean_texts), normalize=False).astype(np.float64)

        self.mean = X.mean(axis=0, keepdims=True)
        self.std = X.std(axis=0, keepdims=True)
        self.std[self.std < self.eps] = 1.0

        Xs = (X - self.mean) / self.std
        cov = np.cov(Xs, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)

        self.eigvals = eigvals[order]
        self.eigvecs = eigvecs[:, order]

        clean_scores = self._score_from_standardized(Xs)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def _score_from_standardized(self, Zs: np.ndarray) -> np.ndarray:
        assert self.eigvals is not None and self.eigvecs is not None
        M_eff = min(self.M, self.eigvals.shape[0])

        scores = np.zeros(Zs.shape[0], dtype=np.float64)
        for i in range(M_eff):
            lam = float(self.eigvals[i])
            lam = lam if lam > self.eps else self.eps
            v = self.eigvecs[:, i]
            scores += np.abs(Zs @ v) / np.sqrt(lam)
        return scores

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        Z = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        Zs = (Z - self.mean) / self.std
        return self._score_from_standardized(Zs)