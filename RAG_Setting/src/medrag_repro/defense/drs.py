from __future__ import annotations

from typing import Sequence

import numpy as np
from drs_defense.core import DRSModel, drs_score, fit_drs

from rag_defenses.common import BaseDetector
from medrag_repro.retriever.contriever import ContrieverEncoder


class DRSDetector(BaseDetector):
    """DRS defense (paper Algorithm 1 & 2); math delegated to drs_defense.core."""

    def __init__(self, encoder: ContrieverEncoder, M: int = 100, clean_quantile: float = 0.99, eps: float = 1e-8):
        super().__init__(two_sided=False, upper_quantile=clean_quantile)
        self.encoder = encoder
        self.M = M
        self.eps = eps
        self.model: DRSModel | None = None
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        X = self.encoder.encode(list(clean_texts), normalize=False).astype(np.float64)
        self.model = fit_drs(X, num_directions=self.M, eps=self.eps)
        self.clean_scores = drs_score(X, self.model)
        self.fit_thresholds_from_scores(self.clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self.model is not None
        Z = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return drs_score(Z, self.model)
