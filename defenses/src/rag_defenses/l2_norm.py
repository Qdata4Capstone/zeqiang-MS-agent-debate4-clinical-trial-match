from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from rag_defenses.common import BaseDetector


def l2_norm_score(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    return np.linalg.norm(embeddings, axis=1)


class L2NormDetector(BaseDetector):
    def __init__(self, encoder, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        super().__init__(two_sided=True, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        self.encoder = encoder
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        X = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return l2_norm_score(X)


def l2_norm_scores(embeddings: torch.Tensor) -> torch.Tensor:
    device = embeddings.device
    scores_np = l2_norm_score(embeddings.detach().cpu().float().numpy())
    return torch.from_numpy(scores_np).float().to(device)
