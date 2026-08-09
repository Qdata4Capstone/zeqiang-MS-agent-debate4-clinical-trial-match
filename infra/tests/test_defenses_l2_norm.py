from __future__ import annotations

import numpy as np

from rag_infra.defenses.l2_norm import l2_norm_score


def test_l2_norm_score_computes_row_wise_euclidean_norm():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])

    result = l2_norm_score(X)

    np.testing.assert_allclose(result, [5.0, 0.0, 1.0])


def test_l2_norm_score_handles_1d_input_as_a_single_row():
    x = np.array([3.0, 4.0])

    result = l2_norm_score(x)

    assert result.shape == (1,)
    np.testing.assert_allclose(result, [5.0])
