from __future__ import annotations

import numpy as np
import pytest
import torch

from rag_defenses.l2_norm import l2_norm_score as core_l2_norm_score
from ReAct.defense_baselines import l2_norm_scores


def test_l2_norm_scores_matches_rag_infra_l2_norm_score_on_a_batch():
    rng = np.random.default_rng(0)
    embeddings_np = rng.normal(size=(20, 6)).astype(np.float32)
    embeddings_t = torch.from_numpy(embeddings_np)

    result = l2_norm_scores(embeddings_t)
    expected = core_l2_norm_score(embeddings_np)

    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


def test_l2_norm_scores_handles_a_single_1d_embedding():
    embeddings_t = torch.tensor([3.0, 4.0])

    result = l2_norm_scores(embeddings_t)

    assert result.item() == pytest.approx(5.0)


def test_l2_norm_scores_preserves_input_device_and_dtype_shape():
    embeddings_t = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

    result = l2_norm_scores(embeddings_t)

    assert result.dtype == torch.float32
    assert result.shape == (2,)
