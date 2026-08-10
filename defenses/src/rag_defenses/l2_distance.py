from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from rag_defenses.common import BaseDetector


class L2DistanceDetector(BaseDetector):
    def __init__(self, encoder, clean_quantile: float = 0.99):
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


def l2_distance_scores(embeddings: torch.Tensor, clean_reference: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    distances = torch.cdist(embeddings.float(), clean_reference.float())
    return distances.min(dim=1).values


def leave_one_out_l2_distance_scores(clean_reference: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(clean_reference.float(), clean_reference.float())
    diagonal_mask = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    distances.masked_fill_(diagonal_mask, float("inf"))
    return distances.min(dim=1).values
