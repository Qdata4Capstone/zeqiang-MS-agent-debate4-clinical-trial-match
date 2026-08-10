from __future__ import annotations

import numpy as np
import torch

from rag_defenses.l2_norm import L2NormDetector, l2_norm_score, l2_norm_scores


class _FakeEncoder:
    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def encode(self, texts, normalize: bool = False) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash((self.seed, t))) % (2**32))
            vecs.append(rng.normal(size=self.dim))
        return np.asarray(vecs, dtype=np.float64)


def test_l2_norm_score_computes_row_wise_euclidean_norm():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(l2_norm_score(X), [5.0, 0.0, 1.0])


def test_l2_norm_score_handles_1d_input_as_a_single_row():
    x = np.array([3.0, 4.0])
    result = l2_norm_score(x)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [5.0])


def test_l2_norm_detector_fits_and_detects_outlier_norms():
    encoder = _FakeEncoder(dim=8, seed=0)
    clean_texts = [f"clean-{i}" for i in range(30)]

    det = L2NormDetector(encoder=encoder)
    det.fit(clean_texts)

    X = encoder.encode(clean_texts, normalize=False).astype(np.float64)
    expected = l2_norm_score(X)
    np.testing.assert_allclose(det.clean_scores, expected)
    assert det.two_sided is True


def test_l2_norm_scores_matches_l2_norm_score_on_a_batch():
    rng = np.random.default_rng(0)
    embeddings_np = rng.normal(size=(20, 6)).astype(np.float32)
    embeddings_t = torch.from_numpy(embeddings_np)

    result = l2_norm_scores(embeddings_t)
    expected = l2_norm_score(embeddings_np)

    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


def test_l2_norm_scores_handles_a_single_1d_embedding():
    import pytest

    embeddings_t = torch.tensor([3.0, 4.0])
    result = l2_norm_scores(embeddings_t)
    assert result.item() == pytest.approx(5.0)
