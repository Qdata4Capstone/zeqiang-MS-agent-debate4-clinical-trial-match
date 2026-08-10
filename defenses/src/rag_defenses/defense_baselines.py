from dataclasses import dataclass

import torch


@dataclass
class QuantileStats:
    threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float


@dataclass
class PerplexityStats:
    lower_threshold: float
    upper_threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float


def fit_upper_quantile(clean_scores: torch.Tensor, quantile: float = 0.99) -> QuantileStats:
    threshold = torch.quantile(clean_scores, quantile).item()
    false_positive_rate = (clean_scores > threshold).float().mean().item()
    return QuantileStats(
        threshold=threshold,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
    )


def fit_two_sided_quantile(clean_scores: torch.Tensor, quantile: float = 0.99) -> PerplexityStats:
    tail = max((1 - quantile) / 2, 1e-4)
    lower = torch.quantile(clean_scores, tail).item()
    upper = torch.quantile(clean_scores, 1 - tail).item()
    false_positive_rate = ((clean_scores < lower) | (clean_scores > upper)).float().mean().item()
    return PerplexityStats(
        lower_threshold=lower,
        upper_threshold=upper,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
    )
