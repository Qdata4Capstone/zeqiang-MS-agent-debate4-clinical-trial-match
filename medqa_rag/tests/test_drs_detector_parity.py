from __future__ import annotations

import numpy as np

from drs_defense.core import drs_score, fit_drs
from medrag_repro.defense.drs import DRSDetector


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


def test_drs_detector_matches_drs_defense_core():
    encoder = _FakeEncoder(dim=8, seed=0)
    clean_texts = [f"clean-doc-{i}" for i in range(50)]
    probe_texts = [f"probe-doc-{i}" for i in range(10)]

    detector = DRSDetector(encoder=encoder, M=5, clean_quantile=0.9)
    detector.fit(clean_texts)

    clean_vectors = encoder.encode(clean_texts)
    expected_model = fit_drs(clean_vectors, num_directions=5)
    expected_clean_scores = drs_score(clean_vectors, expected_model)
    np.testing.assert_allclose(detector.clean_scores, expected_clean_scores, rtol=1e-10)

    got_probe_scores = detector.score_texts(probe_texts)
    expected_probe_scores = drs_score(encoder.encode(probe_texts), expected_model)
    np.testing.assert_allclose(got_probe_scores, expected_probe_scores, rtol=1e-10)
