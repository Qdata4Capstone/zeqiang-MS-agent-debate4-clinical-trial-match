import numpy as np
import pytest

from drs_defense.core import drs_score, fit_drs

CLEAN_X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])


def test_drs_score_matches_hand_computed_value():
    # With num_directions=1 (keeping only the low-variance (1,-1) direction,
    # eigenvalue 8/15 -- see Task 1's derivation), the point z=(1,-1) has
    # DRS(z) = sqrt(3/2) exactly:
    #   z_std = (1,-1)/sqrt(2.5); |proj onto (1,-1)/sqrt(2)| = 2/sqrt(5)
    #   DRS = (2/sqrt(5)) / sqrt(8/15) = sqrt(3/2)
    model = fit_drs(CLEAN_X, num_directions=1)
    score = drs_score(np.array([1.0, -1.0]), model)
    assert score == pytest.approx(np.sqrt(1.5), rel=1e-8)


def test_drs_score_batch_matches_single():
    model = fit_drs(CLEAN_X, num_directions=2)
    points = np.array([[1.0, -1.0], [0.5, 0.5], [3.0, 0.0]])
    batch_scores = drs_score(points, model)
    single_scores = np.array([drs_score(p, model) for p in points])
    np.testing.assert_allclose(batch_scores, single_scores, rtol=1e-12)


def test_drs_score_of_mean_is_near_zero():
    model = fit_drs(CLEAN_X, num_directions=2)
    score = drs_score(np.array([0.0, 0.0]), model)
    assert score == pytest.approx(0.0, abs=1e-10)
