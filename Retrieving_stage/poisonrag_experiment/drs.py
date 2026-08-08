from __future__ import annotations

import numpy as np
from drs_defense.core import DRSModel
from drs_defense.core import drs_score as _core_drs_score
from drs_defense.core import low_variance_eigenbasis, standardize


def fit_drs(clean_embeddings, num_directions=32, power=1.0, eps=1e-8):
    """
    Fit a DRS model on clean embeddings.

    DRS emphasizes shifts along low-variance directions in the clean data
    (paper Eq. 3: https://openreview.net/pdf?id=2aL6gcFX7q). `power` is a
    local ablation knob (not part of the paper's formula); leave it at the
    default 1.0 to reproduce Eq. 3 exactly.
    """
    clean_embeddings = np.asarray(clean_embeddings, dtype=np.float32)
    if clean_embeddings.ndim != 2:
        raise ValueError("clean_embeddings must be a 2D array.")
    if len(clean_embeddings) < 2:
        raise ValueError("At least two clean embeddings are required for DRS.")

    standardized, mean, std = standardize(clean_embeddings, eps=eps)
    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions)

    return {
        "mean": mean,
        "std": std,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "power": power,
        "eps": eps,
    }


def drs_score(embedding, drs_model):
    """Compute DRS for a single embedding."""
    if drs_model["power"] == 1.0:
        model = DRSModel(
            mean=drs_model["mean"],
            std=drs_model["std"],
            eigenvalues=drs_model["eigenvalues"],
            eigenvectors=drs_model["eigenvectors"],
            num_directions=drs_model["eigenvectors"].shape[1],
            eps=drs_model["eps"],
        )
        return float(_core_drs_score(embedding, model))

    z = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    z_std = (z - drs_model["mean"]) / drs_model["std"]
    z_std = z_std[0]
    projections = np.abs(z_std @ drs_model["eigenvectors"]) ** drs_model["power"]
    denom = np.sqrt(np.maximum(drs_model["eigenvalues"], drs_model["eps"]))
    return float(np.sum(projections / denom))


def drs_threshold(clean_embeddings, quantile=0.99, num_directions=32, power=1.0):
    """Fit DRS and derive a threshold from clean samples."""
    model = fit_drs(
        clean_embeddings=clean_embeddings,
        num_directions=num_directions,
        power=power,
    )
    clean_scores = [drs_score(embedding, model) for embedding in clean_embeddings]
    threshold = float(np.quantile(clean_scores, quantile))
    return model, clean_scores, threshold
