from __future__ import annotations

import numpy as np
import pytest
import torch

from drs_defense.core import drs_score as core_drs_score
from drs_defense.core import fit_drs as core_fit_drs
from ReAct.drs import fit_drs, score_drs


def test_fit_and_score_drs_match_drs_defense_core():
    rng = np.random.default_rng(0)
    clean_np = rng.normal(size=(40, 6))
    clean_t = torch.from_numpy(clean_np).float()

    stats = fit_drs(clean_t, num_directions=4, quantile=0.9)
    core_model = core_fit_drs(clean_np, num_directions=4)

    np.testing.assert_allclose(stats.eigenvalues.numpy(), core_model.eigenvalues, rtol=1e-4)
    np.testing.assert_allclose(
        np.abs(stats.eigenvectors.numpy()), np.abs(core_model.eigenvectors), rtol=1e-4,
    )

    probe_np = clean_np[0] + 0.05
    probe_t = torch.from_numpy(probe_np).float()
    got = score_drs(probe_t, stats).item()
    expected = float(core_drs_score(probe_np, core_model))
    assert abs(got - expected) < 1e-3


def test_fit_drs_rejects_non_2d_input():
    with pytest.raises(ValueError):
        fit_drs(torch.zeros(5), num_directions=2)
