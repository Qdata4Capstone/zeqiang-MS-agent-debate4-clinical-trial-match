import numpy as np
import pytest

from drs_defense.core import (
    drs_score,
    fit_drs,
    fit_drs_with_threshold,
    is_flagged,
    quantile_threshold,
)


def test_quantile_threshold_basic():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert quantile_threshold(scores, quantile=0.5) == pytest.approx(3.0)


def test_fit_drs_with_threshold_reproduces_manual_pipeline():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(200, 5))

    model, clean_scores, threshold = fit_drs_with_threshold(clean, num_directions=3, quantile=0.9)

    manual_model = fit_drs(clean, num_directions=3)
    np.testing.assert_allclose(model.eigenvalues, manual_model.eigenvalues)
    assert threshold == quantile_threshold(clean_scores, 0.9)


def test_fit_drs_with_threshold_empirical_fpr_on_held_out_clean_data():
    # Algorithm 2 calibration property: the fraction of *held-out* clean
    # samples exceeding tau should track (1 - quantile).
    rng = np.random.default_rng(42)
    fit_data = rng.normal(size=(1000, 5))
    holdout_data = rng.normal(size=(1000, 5))

    model, _, threshold = fit_drs_with_threshold(fit_data, num_directions=3, quantile=0.9)
    holdout_scores = drs_score(holdout_data, model)

    observed_fpr = is_flagged(holdout_scores, threshold).mean()
    assert abs(observed_fpr - 0.1) < 0.05
