from __future__ import annotations

from typing import Sequence

import numpy as np

from medrag_repro.defense.common import BaseDetector
from medrag_repro.retriever.contriever import ContrieverEncoder


class L2NormDetector(BaseDetector):
    def __init__(self, encoder: ContrieverEncoder, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        super().__init__(two_sided=True, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        self.encoder = encoder
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        X = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return np.linalg.norm(X, axis=1)