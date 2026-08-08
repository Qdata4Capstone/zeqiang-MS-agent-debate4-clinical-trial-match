import numpy as np
import pytest

from drs_defense.core import low_variance_eigenbasis, standardize


def test_standardize_zero_mean_unit_population_variance():
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, mean, std = standardize(X)

    np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(mean, [[0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(std, [[np.sqrt(2.5), np.sqrt(2.5)]], atol=1e-12)


def test_low_variance_eigenbasis_selects_smallest_eigenvalue_direction():
    # Constructed so the standardized covariance (np.cov default ddof=1, N=4)
    # is exactly (4/3) * [[1, 0.6], [0.6, 1]], with eigenvalues 32/15 along
    # (1, 1) (high-variance) and 8/15 along (1, -1) (low-variance).
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, _, _ = standardize(X)

    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=1)

    assert eigenvalues.shape == (1,)
    assert eigenvectors.shape == (2, 1)
    np.testing.assert_allclose(eigenvalues[0], 8.0 / 15.0, rtol=1e-8)

    v = eigenvectors[:, 0]
    # low-variance eigenvector must point along (1, -1): equal magnitude, opposite sign
    np.testing.assert_allclose(abs(v[0]), abs(v[1]), rtol=1e-8)
    assert v[0] * v[1] < 0


def test_low_variance_eigenbasis_clips_num_directions_to_dimensionality():
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, _, _ = standardize(X)
    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=1000)
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)


def test_low_variance_eigenbasis_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        low_variance_eigenbasis(np.zeros((1, 3)), num_directions=1)


def test_low_variance_eigenbasis_requires_positive_num_directions():
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, _, _ = standardize(X)
    with pytest.raises(ValueError):
        low_variance_eigenbasis(standardized, num_directions=0)
