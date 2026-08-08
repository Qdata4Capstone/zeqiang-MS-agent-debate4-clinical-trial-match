import numpy as np
import pytest

from drs_defense.core import drs_score, fit_drs, fit_drs_with_threshold, is_flagged

# Same 2x2 construction as test_core_eigenbasis.py: standardized covariance is
# (4/3) * [[1, 0.6], [0.6, 1]], eigenvalues 32/15 (along (1,1), high-variance)
# and 8/15 (along (1,-1), low-variance).
CORRELATED_CLEAN_X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])


def test_equal_magnitude_shift_scores_higher_along_low_variance_direction():
    """Core paper claim (Sec. 3): 'more effective attacks tend to cause larger
    shifts along directions where the variance is low.' A unit shift along the
    low-variance eigendirection must score higher than an equal-magnitude
    shift along the high-variance eigendirection.
    """
    model = fit_drs(CORRELATED_CLEAN_X, num_directions=2)

    high_variance_shift = np.array([1.0, 1.0]) / np.sqrt(2.0)
    low_variance_shift = np.array([1.0, -1.0]) / np.sqrt(2.0)

    high_score = drs_score(high_variance_shift, model)
    low_score = drs_score(low_variance_shift, model)

    assert low_score > high_score
    # Analytically: ratio = sqrt(eigenvalue_high / eigenvalue_low)
    #             = sqrt((32/15) / (8/15)) = 2
    assert low_score == pytest.approx(2.0 * high_score, rel=1e-6)


def test_perturbation_orthogonal_to_clean_manifold_is_detected():
    """Small-scale replication of Fig. 2(d): clean embeddings concentrate near
    a low-dimensional correlated manifold (as real neural embeddings do).
    Perturbations pushed off that manifold, into its low-variance orthogonal
    complement, are exactly what DRS is designed to flag.
    """
    rng = np.random.default_rng(7)
    d, k, n = 20, 3, 500  # d observed dims, k latent factors, n samples

    loading = rng.normal(size=(k, d))

    def sample_clean(n_samples, noise_std=0.05):
        latents = rng.normal(size=(n_samples, k))
        return latents @ loading + rng.normal(scale=noise_std, size=(n_samples, d))

    clean_fit = sample_clean(n)
    clean_holdout = sample_clean(n)

    model, _, threshold = fit_drs_with_threshold(clean_fit, num_directions=d - k, quantile=0.99)
    clean_flag_rate = is_flagged(drs_score(clean_holdout, model), threshold).mean()

    # Find a direction orthogonal to the manifold spanned by `loading` (its
    # right null space) and push clean points off the manifold along it.
    _, _, vt = np.linalg.svd(loading, full_matrices=True)
    off_manifold_direction = vt[k]  # first right-singular vector outside the row space of `loading`

    poisoned = sample_clean(n) + 3.0 * off_manifold_direction
    poison_flag_rate = is_flagged(drs_score(poisoned, model), threshold).mean()

    assert clean_flag_rate < 0.05
    assert poison_flag_rate > 0.9


def test_handles_zero_variance_dimension_without_nan_or_inf():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    X[:, 2] = 5.0  # constant column: zero variance

    model = fit_drs(X, num_directions=4)
    scores = drs_score(X, model)

    assert np.isfinite(scores).all()
    assert model.num_directions == 4
