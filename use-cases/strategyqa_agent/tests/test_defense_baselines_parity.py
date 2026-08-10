from __future__ import annotations

from ReAct.defense_baselines import (
    PerplexityScorer,
    PerplexityStats,
    QuantileStats,
    fit_two_sided_quantile,
    fit_upper_quantile,
    l2_distance_scores,
    l2_norm_scores,
    leave_one_out_l2_distance_scores,
)
from rag_defenses.defense_baselines import PerplexityStats as CorePerplexityStats
from rag_defenses.defense_baselines import QuantileStats as CoreQuantileStats
from rag_defenses.defense_baselines import fit_two_sided_quantile as core_fit_two_sided_quantile
from rag_defenses.defense_baselines import fit_upper_quantile as core_fit_upper_quantile
from rag_defenses.l2_distance import l2_distance_scores as core_l2_distance_scores
from rag_defenses.l2_distance import leave_one_out_l2_distance_scores as core_leave_one_out_l2_distance_scores
from rag_defenses.l2_norm import l2_norm_scores as core_l2_norm_scores
from rag_defenses.perplexity import PerplexityScorer as CorePerplexityScorer


def test_react_defense_baselines_reexports_rag_defenses_exactly():
    assert QuantileStats is CoreQuantileStats
    assert PerplexityStats is CorePerplexityStats
    assert fit_upper_quantile is core_fit_upper_quantile
    assert fit_two_sided_quantile is core_fit_two_sided_quantile
    assert l2_norm_scores is core_l2_norm_scores
    assert l2_distance_scores is core_l2_distance_scores
    assert leave_one_out_l2_distance_scores is core_leave_one_out_l2_distance_scores
    assert PerplexityScorer is CorePerplexityScorer
