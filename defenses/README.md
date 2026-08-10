# rag_defenses

Shared poisoning-defense detector classes and baseline utilities, extracted
from `use-cases/medqa_rag/` and `use-cases/strategyqa_agent/` so they stop being duplicated per
subproject, following the same pattern already used for `drs_defense/`,
`infra/` (`rag_infra`), and `attacks/` (`rag_attacks`).

Unlike `rag_infra.defenses.l2_norm` (Phase 5, math only), this package holds
full Detector classes — subprojects become thin re-exports over it, matching
what Phase 7a already did for the attack generators.

These are the baseline defenses evaluated alongside DRS (see `drs_defense/`)
in the attack/defense showcase described in the [root
README](../README.md#attack--defense-showcase) — `use-cases/medqa_rag/` and
`use-cases/strategyqa_agent/` both compare all four; `use-cases/trial_retrieval/`
evaluates DRS only.

## Code structure

```
defenses/
  src/rag_defenses/
    common.py              # BaseDetector (shared ABC: fit / score_texts / detect / fit_thresholds_from_scores)
    l2_norm.py               # l2_norm_score, L2NormDetector, l2_norm_scores
    l2_distance.py             # L2DistanceDetector, l2_distance_scores, leave_one_out_l2_distance_scores
    perplexity.py                # PerplexityDetector, PerplexityScorer
    defense_baselines.py           # QuantileStats, PerplexityStats, fit_upper_quantile, fit_two_sided_quantile
  tests/                            # pytest suite (perplexity tests mock HF model/tokenizer loading)
```

## Install

```bash
pip install -e ./defenses
```

## Quick start

Pure numpy, no model download needed:

```python
import numpy as np
from rag_defenses.l2_norm import l2_norm_score

embeddings = np.random.randn(10, 768).astype(np.float32)
scores = l2_norm_score(embeddings)  # L2 norm per row
```

## Modules

- `common.py` — `BaseDetector`, the shared ABC (threshold fitting, two-sided
  vs. one-sided detection). No domain coupling.
- `l2_norm.py` — `l2_norm_score` (pure numpy math), `L2NormDetector` (from
  `use-cases/medqa_rag`, encoder-based), `l2_norm_scores` (from `use-cases/strategyqa_agent`,
  torch-based). All three compute the identical L2-norm formula.
- `l2_distance.py` — `L2DistanceDetector` (from `use-cases/medqa_rag`,
  centroid-distance) and `l2_distance_scores`/`leave_one_out_l2_distance_scores`
  (from `use-cases/strategyqa_agent`, nearest-neighbor-distance). **Two different
  formulas, not duplicates** — confirmed during Phase 5's research — kept
  as distinct names in this one file.
- `perplexity.py` — `PerplexityDetector` (from `use-cases/medqa_rag`) and
  `PerplexityScorer` (from `use-cases/strategyqa_agent`). Both compute the same core
  value (`exp(causal-LM loss)`) but are kept as **two distinct classes, not
  merged** — no phase has done the behavioral-parity work to prove they're
  interchangeable (different defaults, different call patterns).
- `defense_baselines.py` — `QuantileStats`, `PerplexityStats`,
  `fit_upper_quantile`, `fit_two_sided_quantile` (from `use-cases/strategyqa_agent`),
  generic threshold-fitting utilities with no L2/perplexity-specific logic.

`ContrieverEncoder` type hints on the moved classes were loosened to
duck-typing (no import of `medrag_repro.retriever.contriever` here) — this
package does not depend backward on any subproject.

`drs_defense/` and the three DRS adapters
(`use-cases/medqa_rag/.../defense/drs.py`, `use-cases/strategyqa_agent/ReAct/drs.py`,
`use-cases/trial_retrieval/poisonrag_experiment/drs.py`) are untouched by this
extraction — `drs_defense/` isn't moving. `use-cases/medqa_rag`'s `drs.py` only
needed its `BaseDetector` import line repointed here, since `DRSDetector`
extends it.

## Tests

```bash
pip install -e defenses
pytest defenses/tests -q
```

Perplexity-related tests mock `AutoTokenizer.from_pretrained`/
`AutoModelForCausalLM.from_pretrained` — no real model download or network
access is needed to run this suite.
