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
    min(num_directions, d) columns.
    """
    if standardized.shape[0] < 2:
        raise ValueError("At least 2 samples are required to estimate a covariance matrix.")
    if num_directions < 1:
        raise ValueError(f"num_directions must be >= 1, got {num_directions}")
    cov = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    m = min(num_directions, eigenvectors.shape[1])
    return eigenvalues[:m], eigenvectors[:, :m]


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
