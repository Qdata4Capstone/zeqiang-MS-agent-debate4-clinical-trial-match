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
    cov = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    m = min(num_directions, eigenvectors.shape[1])
    return eigenvalues[:m], eigenvectors[:, :m]
