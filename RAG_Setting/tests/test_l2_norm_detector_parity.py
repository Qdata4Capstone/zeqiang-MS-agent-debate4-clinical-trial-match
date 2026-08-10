from __future__ import annotations

import numpy as np

from medrag_repro.defense.l2_norm import L2NormDetector
from rag_defenses.l2_norm import l2_norm_score


class _FakeEncoder:
    """Deterministic offline stand-in for ContrieverEncoder: text -> fixed vector."""

    def __init__(self, dim: int, seed: int = 0):
        self.dim = dim
        self.seed = seed

    def encode(self, texts, normalize: bool = False) -> np.ndarray:
        vecs = []
        for t in texts:
            rng = np.random.default_rng(abs(hash((self.seed, t))) % (2**32))
            vecs.append(rng.normal(size=self.dim))
        return np.asarray(vecs, dtype=np.float64)


def test_l2_norm_detector_matches_rag_infra_l2_norm_score():
    encoder = _FakeEncoder(dim=8, seed=0)
    texts = [f"doc-{i}" for i in range(10)]

    detector = L2NormDetector(encoder=encoder)
    scores = detector.score_texts(texts)

    X = encoder.encode(texts, normalize=False).astype(np.float64)
    expected = l2_norm_score(X)

    np.testing.assert_allclose(scores, expected)
