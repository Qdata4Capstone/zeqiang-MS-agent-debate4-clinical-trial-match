from __future__ import annotations

from typing import Sequence

import numpy as np

from medrag_repro.defense.common import BaseDetector
from medrag_repro.retriever.contriever import ContrieverEncoder


class L2DistanceDetector(BaseDetector):
    def __init__(self, encoder: ContrieverEncoder, clean_quantile: float = 0.99):
        super().__init__(two_sided=False, upper_quantile=clean_quantile)
        self.encoder = encoder
        self.centroid: np.ndarray | None = None
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        X = self.encoder.encode(list(clean_texts), normalize=False).astype(np.float64)
        self.centroid = X.mean(axis=0, keepdims=True)
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self.centroid is not None
        X = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return np.linalg.norm(X - self.centroid, axis=1)