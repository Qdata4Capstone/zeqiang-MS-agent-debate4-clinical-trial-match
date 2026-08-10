from __future__ import annotations

from medrag_repro.defense.common import BaseDetector
from medrag_repro.defense.l2_distance import L2DistanceDetector
from medrag_repro.defense.l2_norm import L2NormDetector
from medrag_repro.defense.perplexity import PerplexityDetector
from rag_defenses.common import BaseDetector as CoreBaseDetector
from rag_defenses.l2_distance import L2DistanceDetector as CoreL2DistanceDetector
from rag_defenses.l2_norm import L2NormDetector as CoreL2NormDetector
from rag_defenses.perplexity import PerplexityDetector as CorePerplexityDetector


def test_medrag_repro_defenses_reexport_rag_defenses_exactly():
    assert BaseDetector is CoreBaseDetector
    assert L2NormDetector is CoreL2NormDetector
    assert L2DistanceDetector is CoreL2DistanceDetector
    assert PerplexityDetector is CorePerplexityDetector
