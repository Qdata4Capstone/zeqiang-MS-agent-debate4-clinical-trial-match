from __future__ import annotations

import numpy as np
import torch

from rag_defenses.l2_distance import (
    L2DistanceDetector,
    l2_distance_scores,
    leave_one_out_l2_distance_scores,
)


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


def test_l2_distance_detector_scores_distance_to_centroid():
    encoder = _FakeEncoder(dim=8, seed=0)
    clean_texts = [f"clean-{i}" for i in range(30)]

    det = L2DistanceDetector(encoder=encoder)
    det.fit(clean_texts)

    X = encoder.encode(clean_texts, normalize=False).astype(np.float64)
    expected_centroid = X.mean(axis=0, keepdims=True)
    np.testing.assert_allclose(det.centroid, expected_centroid)

    expected_scores = np.linalg.norm(X - expected_centroid, axis=1)
    np.testing.assert_allclose(det.clean_scores, expected_scores)
    assert det.two_sided is False


def test_l2_distance_scores_returns_min_distance_to_nearest_reference_point():
    reference = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
    embeddings = torch.tensor([[1.0, 0.0], [9.0, 10.0]])

    result = l2_distance_scores(embeddings, reference)

    np.testing.assert_allclose(result.numpy(), [1.0, 1.0], atol=1e-5)


def test_l2_distance_scores_handles_a_single_1d_embedding():
    reference = torch.tensor([[0.0, 0.0]])
    embedding = torch.tensor([3.0, 4.0])

    result = l2_distance_scores(embedding, reference)

    assert result.item() == 5.0


def test_leave_one_out_l2_distance_scores_excludes_self_distance():
    clean_reference = torch.tensor([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])

    result = leave_one_out_l2_distance_scores(clean_reference)

    # point 0's nearest OTHER point is either point 1 or 2, both at distance 5
    assert result[0].item() == 5.0
    # points 1 and 2 are identical, so each other's nearest neighbor at distance 0
    assert result[1].item() == 0.0
    assert result[2].item() == 0.0
