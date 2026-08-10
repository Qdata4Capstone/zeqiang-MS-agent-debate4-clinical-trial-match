from __future__ import annotations

from dataclasses import dataclass, field

import torch

from drs_defense.core import DRSModel
from drs_defense.core import drs_score as _drs_score_np
from drs_defense.core import fit_drs_with_threshold


@dataclass
class DRSStats:
    mean: torch.Tensor
    std: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float
    num_directions: int
    _np_model: DRSModel = field(repr=False)


def fit_drs(clean_embeddings: torch.Tensor, num_directions: int = 200, quantile: float = 0.99) -> DRSStats:
    if clean_embeddings.dim() != 2:
        raise ValueError("clean_embeddings must be a 2D tensor")

    device = clean_embeddings.device
    clean_np = clean_embeddings.detach().cpu().double().numpy()

    model, clean_scores_np, threshold = fit_drs_with_threshold(
        clean_np, num_directions=num_directions, quantile=quantile,
    )

    return DRSStats(
        mean=torch.from_numpy(model.mean).float().to(device),
        std=torch.from_numpy(model.std).float().to(device),
        eigenvectors=torch.from_numpy(model.eigenvectors).float().to(device),
        eigenvalues=torch.from_numpy(model.eigenvalues).float().to(device),
        threshold=threshold,
        clean_scores=torch.from_numpy(clean_scores_np).float().to(device),
        false_positive_rate=float((clean_scores_np > threshold).mean()),
        num_directions=model.num_directions,
        _np_model=model,
    )


def score_drs(embeddings: torch.Tensor, stats: DRSStats) -> torch.Tensor:
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    device = embeddings.device
    scores_np = _drs_score_np(embeddings.detach().cpu().double().numpy(), stats._np_model)
    return torch.from_numpy(scores_np).float().to(device)
