from __future__ import annotations

import numpy as np

from drs_defense.core import drs_score as core_drs_score
from drs_defense.core import fit_drs as core_fit_drs
from poisonrag_experiment.drs import drs_score, drs_threshold, fit_drs


def _clean_embeddings(seed=0, n=60, d=6):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)).astype(np.float32)


def test_fit_drs_and_score_match_core_at_default_power():
    clean = _clean_embeddings()
    model = fit_drs(clean, num_directions=4)  # power defaults to 1.0
    core_model = core_fit_drs(clean, num_directions=4)

    np.testing.assert_allclose(model["eigenvalues"], core_model.eigenvalues, rtol=1e-6)
    np.testing.assert_allclose(model["eigenvectors"], core_model.eigenvectors, rtol=1e-6)

    probe = clean[0] + 0.1
    np.testing.assert_allclose(
        drs_score(probe, model),
        float(core_drs_score(probe, core_model)),
        rtol=1e-6,
    )


def test_drs_threshold_matches_quantile_of_its_own_clean_scores():
    clean = _clean_embeddings(seed=1)
    model, clean_scores, threshold = drs_threshold(clean, quantile=0.9, num_directions=4)
    assert threshold == np.quantile(clean_scores, 0.9)


def test_drs_score_still_supports_non_default_power_ablation():
    clean = _clean_embeddings(seed=2)
    model = fit_drs(clean, num_directions=4, power=2.0)
    score = drs_score(clean[0], model)
    assert isinstance(score, float)
    assert score >= 0.0
