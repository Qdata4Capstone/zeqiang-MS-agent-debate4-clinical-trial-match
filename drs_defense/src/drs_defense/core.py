"""Reference implementation of the DRS (Directional Relative Shifts) defense.

Source: "Understanding Data Poisoning Attacks for RAG: Insights and Algorithms"
(ICLR 2025 submission), https://openreview.net/pdf?id=2aL6gcFX7q, Section 4,
Algorithm 1 (Compute Directional Relevance Score) and Algorithm 2 (Detection
with DRS).

Algorithm 1 (Eq. 3):
    Given a standardized clean data matrix X in R^(n x d):
    1. Eigendecompose the covariance S = V * Lambda * V^T
    2. Sort eigenvalues (and eigenvectors) ascending: lambda_sigma(1) <= ... <= lambda_sigma(d)
    3. DRS(z; X) = sum_{i=1}^{M} |z^T v_sigma(i)| / sqrt(lambda_sigma(i))
       over the M *smallest*-eigenvalue ("low-variance") directions.

Algorithm 2:
    Fit DRS on top-K retrieved clean documents per protected query, set the
    decision threshold tau to the q-th quantile of the clean DRS scores, and
    reject any future document z with DRS(z; X_clean) > tau.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EPS = 1e-8


def standardize(embeddings, eps: float = DEFAULT_EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise zero-mean, unit-variance standardization.

    Returns (standardized, mean, std); mean/std are shaped (1, d) so they
    broadcast against any (n, d) batch.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D (n, d), got shape {X.shape}")
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std, mean, std


def low_variance_eigenbasis(standardized: np.ndarray, num_directions: int) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1, steps 1-2: eigendecompose the covariance of standardized
    data and keep the `num_directions` eigenvectors with the *smallest*
    eigenvalues.

    Returns (eigenvalues, eigenvectors), ascending, truncated to
    min(num_directions, effective_rank) columns.
    """
    n, d = standardized.shape
    if n < 2:
        raise ValueError("At least 2 samples are required to estimate a covariance matrix.")
    if num_directions < 1:
        raise ValueError(f"num_directions must be >= 1, got {num_directions}")

    if n <= d:
        return _low_variance_eigenbasis_dual(standardized, num_directions)

    cov = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    m = min(num_directions, eigenvectors.shape[1])
    return eigenvalues[:m], eigenvectors[:, :m]


def _low_variance_eigenbasis_dual(standardized: np.ndarray, num_directions: int) -> tuple[np.ndarray, np.ndarray]:
    """Dual/Gram-matrix PCA for the n <= d regime (fewer clean reference
    samples than embedding dimensions -- e.g. drs_ref_k=20 clean docs
    against a 768-dim MedCPT/Contriever embedding).

    Eigendecomposing the full d x d covariance in this regime produces
    d - rank(X) *exact* zero eigenvalues -- directions the n samples have
    literally zero support in, not genuinely low-variance ones -- and
    Algorithm 1's "smallest eigenvalue" selection picks these up first. Eq.
    3's 1/sqrt(eigenvalue) term then explodes for any out-of-sample point
    along them: a clean reference point scores ~0 there (its own covariance
    was fit to make that exactly true), while literally any other point --
    poisoned or legitimately clean -- scores enormous, since nothing
    constrained it to lie in the tiny n-1-dimensional subspace the
    reference set actually spans. This isn't a corner case: with a small
    `ref_k` (or any n <= d) it is the *default* outcome, confirmed against
    a real MedCPT-embedding run (n=20, d=768) where clean scores landed
    around 1e-10 and unseen-candidate scores around 1e5.

    Instead, eigendecompose the n x n Gram matrix X @ X.T: X @ X.T and
    X.T @ X share the same nonzero eigenvalues (see e.g. the dual-PCA
    technique at stats.stackexchange.com/questions/7111), so this recovers
    the exact same low-variance directions the covariance would have given
    -- but only the data's true rank-many of them (<= n-1, since
    mean-centering removes one degree of freedom), with no spurious zeros
    mixed in. If X = U S V^T (SVD), then X @ X.T = U S^2 U^T and
    X.T @ X = V S^2 V^T -- same eigenvalues S^2, and V = X^T @ U / S.
    """
    n = standardized.shape[0]
    gram = (standardized @ standardized.T) / (n - 1)
    dual_eigenvalues, dual_eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(dual_eigenvalues)
    dual_eigenvalues = dual_eigenvalues[order]
    dual_eigenvectors = dual_eigenvectors[:, order]

    # Effective rank: eigenvalues meaningfully above float noise relative to
    # the largest one (mirrors numpy.linalg.matrix_rank's default tolerance),
    # so the trivial exact-zero eigenvalue mean-centering always introduces
    # (and any additional redundancy, e.g. duplicate reference embeddings)
    # gets excluded rather than treated as a "real" low-variance direction.
    tol = dual_eigenvalues[-1] * max(gram.shape) * np.finfo(np.float64).eps
    support = dual_eigenvalues > tol
    dual_eigenvalues = dual_eigenvalues[support]
    dual_eigenvectors = dual_eigenvectors[:, support]

    m = min(num_directions, dual_eigenvectors.shape[1])
    dual_eigenvalues = dual_eigenvalues[:m]
    dual_eigenvectors = dual_eigenvectors[:, :m]

    if m == 0:
        return dual_eigenvalues, np.zeros((standardized.shape[1], 0))

    eigenvectors = (standardized.T @ dual_eigenvectors) / np.sqrt(dual_eigenvalues * (n - 1))
    eigenvectors /= np.linalg.norm(eigenvectors, axis=0, keepdims=True)
    return dual_eigenvalues, eigenvectors


@dataclass(frozen=True)
class DRSModel:
    """Fitted DRS reference model: mean/std/eigenbasis of the clean data."""

    mean: np.ndarray
    std: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    num_directions: int
    eps: float = DEFAULT_EPS


def fit_drs(clean_embeddings, num_directions: int = 100, eps: float = DEFAULT_EPS) -> DRSModel:
    """Fit a DRS model on clean embeddings (Algorithm 1, steps 1-2)."""
    standardized, mean, std = standardize(clean_embeddings, eps=eps)
    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions)
    return DRSModel(
        mean=mean,
        std=std,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        num_directions=eigenvectors.shape[1],
        eps=eps,
    )


def drs_score(embeddings, model: DRSModel):
    """Eq. 3: DRS(z; X) = sum_i |z^T v_i| / sqrt(lambda_i).

    Accepts a single embedding (d,) or a batch (n, d); returns a scalar
    (np.float64) for the single case, or a (n,) array for the batch case.
    """
    Z = np.asarray(embeddings, dtype=np.float64)
    single = Z.ndim == 1
    if single:
        Z = Z[None, :]
    standardized = (Z - model.mean) / model.std
    projections = np.abs(standardized @ model.eigenvectors)
    scales = np.sqrt(np.maximum(model.eigenvalues, model.eps))
    scores = (projections / scales).sum(axis=1)
    return scores[0] if single else scores


def quantile_threshold(scores, quantile: float = 0.99) -> float:
    """Algorithm 2, step 4: tau = q-th quantile of clean DRS scores."""
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))


def fit_drs_with_threshold(
    clean_embeddings,
    num_directions: int = 100,
    quantile: float = 0.99,
    eps: float = DEFAULT_EPS,
) -> tuple[DRSModel, np.ndarray, float]:
    """Algorithm 2, steps 1-4: fit + score clean data + derive tau."""
    model = fit_drs(clean_embeddings, num_directions=num_directions, eps=eps)
    clean_scores = drs_score(clean_embeddings, model)
    threshold = quantile_threshold(clean_scores, quantile)
    return model, clean_scores, threshold


def is_flagged(scores, threshold: float) -> np.ndarray:
    """Algorithm 2, step 5: reject z if DRS(z; X_clean) > tau."""
    return np.asarray(scores) > threshold
