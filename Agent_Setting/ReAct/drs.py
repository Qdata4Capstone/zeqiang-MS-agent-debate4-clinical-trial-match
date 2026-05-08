from dataclasses import dataclass

import torch


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


def _standardize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std.clamp_min(1e-6)


def score_drs(embeddings: torch.Tensor, stats: DRSStats) -> torch.Tensor:
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    z = _standardize(embeddings, stats.mean, stats.std)
    projections = torch.abs(z @ stats.eigenvectors[:, : stats.num_directions])
    scales = torch.sqrt(stats.eigenvalues[: stats.num_directions].clamp_min(1e-8))
    return (projections / scales).sum(dim=1)


def fit_drs(clean_embeddings: torch.Tensor, num_directions: int = 200, quantile: float = 0.99) -> DRSStats:
    if clean_embeddings.dim() != 2:
        raise ValueError("clean_embeddings must be a 2D tensor")

    clean_embeddings = clean_embeddings.float()
    mean = clean_embeddings.mean(dim=0)
    std = clean_embeddings.std(dim=0, unbiased=False)
    standardized = _standardize(clean_embeddings, mean, std)

    cov = torch.cov(standardized.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    max_directions = min(num_directions, eigenvectors.shape[1])
    trimmed_vectors = eigenvectors[:, :max_directions]
    trimmed_values = eigenvalues[:max_directions]

    clean_scores = (
        torch.abs(standardized @ trimmed_vectors) / torch.sqrt(trimmed_values.clamp_min(1e-8))
    ).sum(dim=1)
    threshold = torch.quantile(clean_scores, quantile).item()
    false_positive_rate = (clean_scores > threshold).float().mean().item()

    return DRSStats(
        mean=mean,
        std=std,
        eigenvectors=eigenvectors,
        eigenvalues=eigenvalues,
        threshold=threshold,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
        num_directions=max_directions,
    )
