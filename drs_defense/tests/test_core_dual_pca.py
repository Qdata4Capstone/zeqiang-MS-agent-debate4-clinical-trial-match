"""Regression tests for the dual (Gram-matrix) PCA path in
low_variance_eigenbasis, used when n <= d (fewer clean reference samples
than embedding dimensions).

Background: a real run of trial_retrieval's poisonrag_experiment against
768-dim MedCPT embeddings with only --drs_ref_k 20 clean reference
documents (n=20, d=768) found DRS flagging 55-72% of an entire 3.6k-doc
corpus as poisoned. Tracing it: eigendecomposing the full d x d covariance
when n <= d produces d - rank(X) *exact* zero eigenvalues (directions the
n samples have literally zero support in), and Algorithm 1's "smallest
eigenvalue" selection picks these up first. Clean reference points scored
~1e-10 on them (by construction: a zero-eigenvalue direction of a sample
covariance has zero variance across exactly those samples), while any
other point -- poisoned or legitimately clean -- scored ~1e5, since
nothing constrained it to the reference set's tiny actual span. See
stats.stackexchange.com/questions/7111 for the dual-PCA technique this
fixes it with: eigendecompose the n x n Gram matrix X @ X.T instead of the
d x d covariance X.T @ X -- they share the same nonzero eigenvalues, but
the n x n form caps the number of directions at the data's true rank
(<= n-1) with no spurious zeros mixed in.
"""
import numpy as np

from drs_defense.core import drs_score, fit_drs, low_variance_eigenbasis, standardize


def test_dual_pca_matches_primal_covariance_nonzero_spectrum():
    """When n <= d, low_variance_eigenbasis must route to the dual
    formulation and recover eigenvalues/eigenvectors matching the *real*
    (non-spurious) part of the full covariance's spectrum exactly -- not
    an approximation, an exact shared-eigenvalue identity."""
    rng = np.random.default_rng(11)
    n, d = 8, 30  # n < d: the regime that used to be broken
    standardized, _, _ = standardize(rng.normal(size=(n, d)))

    # Independently eigendecompose the full covariance for comparison.
    cov = np.cov(standardized, rowvar=False)
    primal_eigenvalues, primal_eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(primal_eigenvalues)
    primal_eigenvalues = primal_eigenvalues[order]
    primal_eigenvectors = primal_eigenvectors[:, order]

    rank = n - 1  # mean-centered n x d data has rank <= n - 1
    # The d - rank smallest primal eigenvalues are spurious exact zeros;
    # the remaining `rank` ascending ones are the real, data-supported ones.
    real_primal_eigenvalues = primal_eigenvalues[d - rank:]
    real_primal_eigenvectors = primal_eigenvectors[:, d - rank:]

    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=rank)

    assert eigenvectors.shape == (d, rank)
    np.testing.assert_allclose(eigenvalues, real_primal_eigenvalues, rtol=1e-6, atol=1e-8)

    for i in range(rank):
        np.testing.assert_allclose(np.linalg.norm(eigenvectors[:, i]), 1.0, atol=1e-8)
        matches_up_to_sign = np.allclose(
            eigenvectors[:, i], real_primal_eigenvectors[:, i], atol=1e-6
        ) or np.allclose(eigenvectors[:, i], -real_primal_eigenvectors[:, i], atol=1e-6)
        assert matches_up_to_sign, f"eigenvector {i} doesn't match the primal one (up to sign)"


def test_dual_pca_caps_directions_at_true_rank_not_dimensionality():
    """Requesting more directions than the data can support (n - 1) must
    clip to the true rank, not silently substitute spurious exact-zero
    covariance eigenvalues the way the pre-fix code did."""
    rng = np.random.default_rng(5)
    n, d = 10, 100
    standardized, _, _ = standardize(rng.normal(size=(n, d)))

    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=1000)

    assert eigenvectors.shape == (d, n - 1)
    assert eigenvalues.shape == (n - 1,)
    # None of the kept eigenvalues should be anywhere near the machine-noise
    # floor a spurious zero eigenvalue would sit at.
    assert eigenvalues.min() > 1e-6


def test_no_catastrophic_score_blowup_when_reference_set_smaller_than_dimensionality():
    """Regression test for the real bug: with the fix, an unseen point's
    DRS score must stay within a sane multiple of the clean reference
    scores' own spread, not many orders of magnitude larger. Before the
    fix, this exact shape (n=20, d=200) produced clean scores ~1e-10 and
    unseen scores ~1e5 -- a ~1e14 ratio. After the fix, unseen points here
    (drawn from the identical distribution as the "clean" ones, so there's
    no real distributional shift to detect) score *lower* on average than
    the training points, confirming the fictitious directions are gone."""
    rng = np.random.default_rng(3)
    n, d = 20, 200
    clean = rng.normal(size=(n, d))

    model = fit_drs(clean, num_directions=16)
    assert model.num_directions == 16  # requested directions were satisfiable (16 < n-1=19)

    clean_scores = drs_score(clean, model)
    unseen_scores = drs_score(rng.normal(size=(50, d)), model)

    assert np.isfinite(clean_scores).all()
    assert np.isfinite(unseen_scores).all()
    # Before the fix this ratio was ~1e14; same-distribution unseen points
    # should land within a small constant factor of the clean spread, not
    # explode.
    assert unseen_scores.max() < clean_scores.max() * 10


def test_dual_pca_handles_num_directions_exceeding_rank_gracefully():
    """A degenerate n=2 reference set (rank 1) requesting many directions
    must not crash or divide by a filtered-out zero eigenvalue."""
    rng = np.random.default_rng(9)
    standardized, _, _ = standardize(rng.normal(size=(2, 50)))

    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=16)

    assert eigenvectors.shape == (50, 1)
    assert eigenvalues.shape == (1,)
    assert np.isfinite(eigenvalues).all()
