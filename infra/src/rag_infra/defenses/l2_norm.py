from __future__ import annotations

import numpy as np


def l2_norm_score(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    return np.linalg.norm(embeddings, axis=1)
