from __future__ import annotations

import torch

from rag_defenses.defense_baselines import (
    PerplexityStats,
    QuantileStats,
    fit_two_sided_quantile,
    fit_upper_quantile,
)


def test_fit_upper_quantile_computes_threshold_and_false_positive_rate():
    clean_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    result = fit_upper_quantile(clean_scores, quantile=0.8)

    assert isinstance(result, QuantileStats)
    assert result.threshold == torch.quantile(clean_scores, 0.8).item()
    assert result.false_positive_rate == (clean_scores > result.threshold).float().mean().item()


def test_fit_two_sided_quantile_computes_symmetric_tail_thresholds():
    clean_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    result = fit_two_sided_quantile(clean_scores, quantile=0.8)

    assert isinstance(result, PerplexityStats)
    tail = (1 - 0.8) / 2
    assert result.lower_threshold == torch.quantile(clean_scores, tail).item()
    assert result.upper_threshold == torch.quantile(clean_scores, 1 - tail).item()


def test_fit_two_sided_quantile_clamps_tiny_tail_to_minimum():
    clean_scores = torch.tensor([1.0, 2.0, 3.0])

    result = fit_two_sided_quantile(clean_scores, quantile=0.9999)

    # tail = (1-0.9999)/2 = 0.00005, clamped to 1e-4
    expected_tail = 1e-4
    assert result.lower_threshold == torch.quantile(clean_scores, expected_tail).item()
