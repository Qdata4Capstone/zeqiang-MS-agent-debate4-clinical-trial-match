# defenses/ Package Extraction (Phase 7b) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a new top-level `defenses/` package (dist name `rag-defenses`, import name `rag_defenses`) holding the full Detector classes and baseline-defense utilities currently split across `RAG_Setting`'s `defense/` directory and `Agent_Setting`'s `ReAct/defense_baselines.py`. Unlike Phase 5 (which extracted only `l2_norm`'s pure math, leaving the Detector classes behind as adapters), this phase relocates the **whole classes** — per explicit user decision, matching the "thin experiment-runner" goal — so both subprojects' `defense/`-area files become pure re-exports, the same pattern Phase 7a just used for the attack generator.

**Architecture:**
- `rag_defenses.common.BaseDetector` — the ABC, moved unchanged (already fully generic, no domain coupling).
- `rag_defenses.l2_norm` — `l2_norm_score` (already here from Phase 5) **plus** the `L2NormDetector` class (moved from `RAG_Setting`) **plus** the `l2_norm_scores` torch function (moved from `Agent_Setting`) — all three genuinely-shared L2-norm implementations now live in one file.
- `rag_defenses.l2_distance` — `L2DistanceDetector` (centroid-distance, moved from `RAG_Setting`) **and** `l2_distance_scores`/`leave_one_out_l2_distance_scores` (nearest-neighbor-distance, moved from `Agent_Setting`) — two different formulas confirmed in Phase 5, kept as distinct names in one file, not merged.
- `rag_defenses.perplexity` — `PerplexityDetector` (moved from `RAG_Setting`) **and** `PerplexityScorer` (moved from `Agent_Setting`) — same core computation (`exp(causal-LM loss)`) but kept as two distinct classes, not merged (consistent with this repo's rule throughout: physical relocation doesn't require merging; only genuinely-proven-identical code gets consolidated, and no phase has done the behavioral-parity work to prove these two are interchangeable).
- `rag_defenses.defense_baselines` — `QuantileStats`, `PerplexityStats`, `fit_upper_quantile`, `fit_two_sided_quantile` (moved from `Agent_Setting`, the remaining generic threshold-fitting utilities with no L2/perplexity-specific logic).
- `L2DistanceDetector`/`L2NormDetector`/`PerplexityDetector` currently type-hint `encoder: ContrieverEncoder` — since `ContrieverEncoder` is `RAG_Setting`-owned and `rag_defenses` must not depend backward on a subproject, these type hints are loosened (duck-typed, no import of `ContrieverEncoder`) — this is a type-hint-only change, `self.encoder.encode(...)` call sites are unchanged.
- `RAG_Setting/src/medrag_repro/defense/drs.py` (a DRS adapter over `drs_defense/`, itself untouched by this phase per the design spec) extends `BaseDetector` — since `BaseDetector` physically moves, `drs.py`'s **import line** for `BaseDetector` must repoint to `rag_defenses.common`. This is the one necessary touch to an otherwise-untouched file; `DRSDetector`'s own logic/behavior does not change.
- `drs_defense/` itself and all three DRS adapters' *logic* stay exactly as they are — only `medqa_rag`'s (currently `RAG_Setting`'s) `drs.py` import line changes, nothing in `Agent_Setting/ReAct/drs.py` or `Retrieving_stage/poisonrag_experiment/drs.py`.

**Tech Stack:** Python, `numpy`, `torch`, `transformers` (both `PerplexityDetector` and `PerplexityScorer` load a real HuggingFace causal LM — `rag_defenses` is the first shared package taking on this dependency, now viable since both subprojects already have it pinned: `RAG_Setting` requires `torch>=2.1.0`/`transformers>=4.41.0`, `Agent_Setting` pins `torch==2.0.1`/`transformers==4.39.1` exactly — a `torch>=2.0.1`/`transformers>=4.39.1` floor in `rag_defenses`'s `pyproject.toml` satisfies both, matching the boundary exactly). `pytest` with `unittest.mock` (mocking `AutoTokenizer.from_pretrained`/`AutoModelForCausalLM.from_pretrained` in tests — loading a real model would be slow and network-dependent).

## Global Constraints

- `rag_defenses` must stay importable under both Python 3.9 (`Agent_Setting`) and Python 3.10 (`RAG_Setting`) — no syntax newer than 3.9.
- `rag_defenses/pyproject.toml` declares `numpy>=1.24`, `torch>=2.0.1`, `transformers>=4.39.1` as dependencies — no dependency on `medrag-repro`, `poisonrag_experiment`, or any subproject package (matching the established convention: cross-local-package dependencies are wired via each subproject's `requirements.txt`/`environment.yml`, never declared in a shared package's `pyproject.toml`).
- No changes to call-site behavior anywhere: `RAG_Setting/scripts/run_defense.py` (constructs `L2NormDetector`, `L2DistanceDetector`, `PerplexityDetector`, `DRSDetector`) and `Agent_Setting/ReAct/local_wikienv.py` (imports 5 names from `ReAct.defense_baselines`) are NOT touched by this plan and must keep working with unmodified imports.
- `drs_defense/` and the DRS math/behavior in all three adapters (`RAG_Setting/src/medrag_repro/defense/drs.py`, `Agent_Setting/ReAct/drs.py`, `Retrieving_stage/poisonrag_experiment/drs.py`) are untouched — the ONE exception is `RAG_Setting/src/medrag_repro/defense/drs.py`'s `BaseDetector` import line, which must repoint to `rag_defenses.common` since `BaseDetector` physically moves. `DRSDetector`'s class body is otherwise byte-for-byte unchanged.
- `ContrieverEncoder` type hints in the moved classes are loosened (no import of `medrag_repro.retriever.contriever.ContrieverEncoder` in `rag_defenses`) — purely a type-hint change, `self.encoder.encode(texts, normalize=False)` duck-typed calls are unchanged.
- The two perplexity implementations (`PerplexityDetector`, `PerplexityScorer`) are relocated as distinct classes, NOT merged — no phase has proven them behaviorally interchangeable (different default `model_name`/`device` defaults, different quantile-fitting call patterns at their call sites).
- Both `RAG_Setting/requirements.txt` and `Agent_Setting/environment.yml` need a new `-e ../defenses` line (both already have `-e ../drs_defense` and `-e ../infra` from prior phases; `RAG_Setting` also has `-e ../attacks` from Phase 7a).
- This plan does NOT rename `RAG_Setting/` or `Agent_Setting/` — that's Phase 7c, which runs after this plan.
- Run all commands from the `safematch_v3` worktree root: `/Users/qiyanjun/Code/Public/zeqiang-MS-agent-debate4-clinical-trial-match/.worktrees/safematch_v3`.

---

### Task 1: `defenses/` package scaffold + `rag_defenses.common`

**Files:**
- Create: `defenses/pyproject.toml`
- Create: `defenses/README.md`
- Create: `defenses/src/rag_defenses/__init__.py`
- Create: `defenses/src/rag_defenses/common.py`
- Test: `defenses/tests/test_common.py`

**Interfaces:**
- Produces: `rag_defenses.common.BaseDetector` — ABC with `__init__(two_sided=False, upper_quantile=0.99, lower_quantile=0.01)`, abstract `fit(clean_texts)`, abstract `score_texts(texts) -> np.ndarray`, concrete `detect(texts) -> list[bool]`, concrete `fit_thresholds_from_scores(clean_scores) -> None`.
- Produces: the `rag_defenses` package installed editable in the active environment (later tasks assume `import rag_defenses...` resolves).

- [ ] **Step 1: Create the package skeleton**

Create `defenses/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-defenses"
version = "0.1.0"
description = "Shared poisoning-defense detector classes and baseline utilities: BaseDetector, L2-norm/L2-distance/perplexity detectors, and quantile-fitting baselines used across RAG_Setting and Agent_Setting."
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24",
  "torch>=2.0.1",
  "transformers>=4.39.1",
]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

Create `defenses/README.md`:

```markdown
# rag_defenses

Shared poisoning-defense detector classes and baseline utilities, extracted
from `RAG_Setting/` and `Agent_Setting/` so they stop being duplicated per
subproject, following the same pattern already used for `drs_defense/`,
`infra/` (`rag_infra`), and `attacks/` (`rag_attacks`).

Unlike `rag_infra.defenses.l2_norm` (Phase 5, math only), this package holds
full Detector classes — subprojects become thin re-exports over it, matching
what Phase 7a already did for the attack generators.

- `common.py` — `BaseDetector`, the shared ABC (threshold fitting, two-sided
  vs. one-sided detection). No domain coupling.
- `l2_norm.py` — `l2_norm_score` (pure numpy math), `L2NormDetector` (from
  `RAG_Setting`, encoder-based), `l2_norm_scores` (from `Agent_Setting`,
  torch-based). All three compute the identical L2-norm formula.
- `l2_distance.py` — `L2DistanceDetector` (from `RAG_Setting`,
  centroid-distance) and `l2_distance_scores`/`leave_one_out_l2_distance_scores`
  (from `Agent_Setting`, nearest-neighbor-distance). **Two different
  formulas, not duplicates** — confirmed during Phase 5's research — kept
  as distinct names in this one file.
- `perplexity.py` — `PerplexityDetector` (from `RAG_Setting`) and
  `PerplexityScorer` (from `Agent_Setting`). Both compute the same core
  value (`exp(causal-LM loss)`) but are kept as **two distinct classes, not
  merged** — no phase has done the behavioral-parity work to prove they're
  interchangeable (different defaults, different call patterns).
- `defense_baselines.py` — `QuantileStats`, `PerplexityStats`,
  `fit_upper_quantile`, `fit_two_sided_quantile` (from `Agent_Setting`),
  generic threshold-fitting utilities with no L2/perplexity-specific logic.

`ContrieverEncoder` type hints on the moved classes were loosened to
duck-typing (no import of `medrag_repro.retriever.contriever` here) — this
package does not depend backward on any subproject.

`drs_defense/` and the three DRS adapters
(`RAG_Setting/.../defense/drs.py`, `Agent_Setting/ReAct/drs.py`,
`Retrieving_stage/poisonrag_experiment/drs.py`) are untouched by this
extraction — `drs_defense/` isn't moving. `RAG_Setting`'s `drs.py` only
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
```

Create `defenses/src/rag_defenses/__init__.py` (empty file).

- [ ] **Step 2: Install the package editable**

Run: `pip install -e defenses`
Expected: `Successfully installed rag-defenses-0.1.0`

- [ ] **Step 3: Write the failing tests**

Create `defenses/tests/test_common.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from rag_defenses.common import BaseDetector


class _ConstantScoreDetector(BaseDetector):
    """Minimal concrete subclass for testing BaseDetector's shared logic."""

    def __init__(self, scores_by_text, **kwargs):
        super().__init__(**kwargs)
        self.scores_by_text = scores_by_text

    def fit(self, clean_texts):
        pass

    def score_texts(self, texts):
        return np.array([self.scores_by_text[t] for t in texts], dtype=np.float64)


def test_fit_thresholds_from_scores_one_sided_sets_only_upper():
    det = _ConstantScoreDetector({}, two_sided=False, upper_quantile=0.9)
    det.fit_thresholds_from_scores(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert det.lower_threshold is None
    assert det.upper_threshold == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9))


def test_fit_thresholds_from_scores_two_sided_sets_both():
    det = _ConstantScoreDetector({}, two_sided=True, lower_quantile=0.1, upper_quantile=0.9)
    det.fit_thresholds_from_scores(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert det.lower_threshold == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.1))
    assert det.upper_threshold == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9))


def test_detect_one_sided_flags_scores_above_upper_threshold():
    det = _ConstantScoreDetector({"clean": 1.0, "poison": 100.0}, two_sided=False, upper_quantile=0.99)
    det.upper_threshold = 10.0

    assert det.detect(["clean", "poison"]) == [False, True]


def test_detect_two_sided_flags_scores_outside_either_threshold():
    det = _ConstantScoreDetector({"low": -50.0, "mid": 1.0, "high": 50.0}, two_sided=True)
    det.lower_threshold = -10.0
    det.upper_threshold = 10.0

    assert det.detect(["low", "mid", "high"]) == [True, False, True]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest defenses/tests/test_common.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_defenses.common'`

- [ ] **Step 5: Implement `common.py`**

Create `defenses/src/rag_defenses/common.py` (relocated unchanged from `RAG_Setting/src/medrag_repro/defense/common.py`):

```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np


class BaseDetector(ABC):
    def __init__(self, two_sided: bool = False, upper_quantile: float = 0.99, lower_quantile: float = 0.01):
        self.two_sided = two_sided
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.lower_threshold: float | None = None
        self.upper_threshold: float | None = None

    @abstractmethod
    def fit(self, clean_texts: Sequence[str]) -> None:
        ...

    @abstractmethod
    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def detect(self, texts: Sequence[str]) -> list[bool]:
        scores = self.score_texts(texts)
        if self.two_sided:
            assert self.lower_threshold is not None and self.upper_threshold is not None
            return ((scores < self.lower_threshold) | (scores > self.upper_threshold)).tolist()
        else:
            assert self.upper_threshold is not None
            return (scores > self.upper_threshold).tolist()

    def fit_thresholds_from_scores(self, clean_scores: np.ndarray) -> None:
        if self.two_sided:
            self.lower_threshold = float(np.quantile(clean_scores, self.lower_quantile))
            self.upper_threshold = float(np.quantile(clean_scores, self.upper_quantile))
        else:
            self.upper_threshold = float(np.quantile(clean_scores, self.upper_quantile))
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest defenses/tests/test_common.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add defenses/pyproject.toml defenses/README.md defenses/src/rag_defenses/__init__.py defenses/src/rag_defenses/common.py defenses/tests/test_common.py
git commit -m "feat(defenses): add rag_defenses package with common.BaseDetector"
```

---

### Task 2: `rag_defenses.l2_norm` (`l2_norm_score` + `L2NormDetector` + `l2_norm_scores`)

**Files:**
- Create: `defenses/src/rag_defenses/l2_norm.py`
- Test: `defenses/tests/test_l2_norm.py`

**Interfaces:**
- Consumes: `rag_defenses.common.BaseDetector` (Task 1).
- Produces: `rag_defenses.l2_norm.l2_norm_score(embeddings: np.ndarray) -> np.ndarray`, `rag_defenses.l2_norm.L2NormDetector` (class, `__init__(encoder, lower_quantile=0.01, upper_quantile=0.99)`), `rag_defenses.l2_norm.l2_norm_scores(embeddings: torch.Tensor) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

Create `defenses/tests/test_l2_norm.py`:

```python
from __future__ import annotations

import numpy as np
import torch

from rag_defenses.l2_norm import L2NormDetector, l2_norm_score, l2_norm_scores


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


def test_l2_norm_score_computes_row_wise_euclidean_norm():
    X = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(l2_norm_score(X), [5.0, 0.0, 1.0])


def test_l2_norm_score_handles_1d_input_as_a_single_row():
    x = np.array([3.0, 4.0])
    result = l2_norm_score(x)
    assert result.shape == (1,)
    np.testing.assert_allclose(result, [5.0])


def test_l2_norm_detector_fits_and_detects_outlier_norms():
    encoder = _FakeEncoder(dim=8, seed=0)
    clean_texts = [f"clean-{i}" for i in range(30)]

    det = L2NormDetector(encoder=encoder)
    det.fit(clean_texts)

    X = encoder.encode(clean_texts, normalize=False).astype(np.float64)
    expected = l2_norm_score(X)
    np.testing.assert_allclose(det.clean_scores, expected)
    assert det.two_sided is True


def test_l2_norm_scores_matches_l2_norm_score_on_a_batch():
    rng = np.random.default_rng(0)
    embeddings_np = rng.normal(size=(20, 6)).astype(np.float32)
    embeddings_t = torch.from_numpy(embeddings_np)

    result = l2_norm_scores(embeddings_t)
    expected = l2_norm_score(embeddings_np)

    np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)


def test_l2_norm_scores_handles_a_single_1d_embedding():
    import pytest

    embeddings_t = torch.tensor([3.0, 4.0])
    result = l2_norm_scores(embeddings_t)
    assert result.item() == pytest.approx(5.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest defenses/tests/test_l2_norm.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_defenses.l2_norm'`

- [ ] **Step 3: Implement `l2_norm.py`**

Create `defenses/src/rag_defenses/l2_norm.py` (merges `infra/src/rag_infra/defenses/l2_norm.py`'s `l2_norm_score`, `RAG_Setting/src/medrag_repro/defense/l2_norm.py`'s `L2NormDetector` with its `ContrieverEncoder` type hint loosened and `BaseDetector` import repointed, and `Agent_Setting/ReAct/defense_baselines.py`'s `l2_norm_scores`):

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from rag_defenses.common import BaseDetector


def l2_norm_score(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    return np.linalg.norm(embeddings, axis=1)


class L2NormDetector(BaseDetector):
    def __init__(self, encoder, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        super().__init__(two_sided=True, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        self.encoder = encoder
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        X = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return l2_norm_score(X)


def l2_norm_scores(embeddings: torch.Tensor) -> torch.Tensor:
    device = embeddings.device
    scores_np = l2_norm_score(embeddings.detach().cpu().float().numpy())
    return torch.from_numpy(scores_np).float().to(device)
```

Note: `encoder` has no type annotation (was `ContrieverEncoder`, `RAG_Setting`-owned) — `rag_defenses` must not import from any subproject. `self.encoder.encode(...)` usage is unchanged (duck-typed).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest defenses/tests/test_l2_norm.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add defenses/src/rag_defenses/l2_norm.py defenses/tests/test_l2_norm.py
git commit -m "feat(defenses): add rag_defenses.l2_norm (l2_norm_score + L2NormDetector + l2_norm_scores)"
```

---

### Task 3: `rag_defenses.l2_distance` (`L2DistanceDetector` + `l2_distance_scores` + `leave_one_out_l2_distance_scores`)

**Files:**
- Create: `defenses/src/rag_defenses/l2_distance.py`
- Test: `defenses/tests/test_l2_distance.py`

**Interfaces:**
- Consumes: `rag_defenses.common.BaseDetector` (Task 1).
- Produces: `rag_defenses.l2_distance.L2DistanceDetector` (class, `__init__(encoder, clean_quantile=0.99)`), `rag_defenses.l2_distance.l2_distance_scores(embeddings, clean_reference) -> torch.Tensor`, `rag_defenses.l2_distance.leave_one_out_l2_distance_scores(clean_reference) -> torch.Tensor`.

- [ ] **Step 1: Write the failing tests**

Create `defenses/tests/test_l2_distance.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest defenses/tests/test_l2_distance.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_defenses.l2_distance'`

- [ ] **Step 3: Implement `l2_distance.py`**

Create `defenses/src/rag_defenses/l2_distance.py` (relocated from `RAG_Setting/src/medrag_repro/defense/l2_distance.py`'s `L2DistanceDetector`, with `ContrieverEncoder` type hint loosened and `BaseDetector` import repointed, plus `Agent_Setting/ReAct/defense_baselines.py`'s `l2_distance_scores`/`leave_one_out_l2_distance_scores`):

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from rag_defenses.common import BaseDetector


class L2DistanceDetector(BaseDetector):
    def __init__(self, encoder, clean_quantile: float = 0.99):
        super().__init__(two_sided=False, upper_quantile=clean_quantile)
        self.encoder = encoder
        self.centroid: np.ndarray | None = None
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        X = self.encoder.encode(list(clean_texts), normalize=False).astype(np.float64)
        self.centroid = X.mean(axis=0, keepdims=True)
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self.centroid is not None
        X = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return np.linalg.norm(X - self.centroid, axis=1)


def l2_distance_scores(embeddings: torch.Tensor, clean_reference: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    distances = torch.cdist(embeddings.float(), clean_reference.float())
    return distances.min(dim=1).values


def leave_one_out_l2_distance_scores(clean_reference: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(clean_reference.float(), clean_reference.float())
    diagonal_mask = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    distances.masked_fill_(diagonal_mask, float("inf"))
    return distances.min(dim=1).values
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest defenses/tests/test_l2_distance.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add defenses/src/rag_defenses/l2_distance.py defenses/tests/test_l2_distance.py
git commit -m "feat(defenses): add rag_defenses.l2_distance (L2DistanceDetector + l2_distance_scores + leave_one_out_l2_distance_scores)"
```

---

### Task 4: `rag_defenses.perplexity` (`PerplexityDetector` + `PerplexityScorer`)

**Files:**
- Create: `defenses/src/rag_defenses/perplexity.py`
- Test: `defenses/tests/test_perplexity.py`

**Interfaces:**
- Consumes: `rag_defenses.common.BaseDetector` (Task 1).
- Produces: `rag_defenses.perplexity.PerplexityDetector` (class, `__init__(model_name, device="cpu", lower_quantile=0.01, upper_quantile=0.99, max_length=512)`), `rag_defenses.perplexity.PerplexityScorer` (class, `__init__(model_name="gpt2", device="cuda")`).

- [ ] **Step 1: Write the failing tests**

Create `defenses/tests/test_perplexity.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from rag_defenses.perplexity import PerplexityDetector, PerplexityScorer


def _mock_causal_lm(loss_value: float):
    """Build mock tokenizer/model objects matching AutoTokenizer/AutoModelForCausalLM's call shape."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "already-set"
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.loss = torch.tensor(loss_value)
    mock_model.return_value = mock_outputs
    mock_model.to.return_value = mock_model

    return mock_tokenizer, mock_model


def test_perplexity_detector_fits_and_scores_from_causal_lm_loss():
    mock_tokenizer, mock_model = _mock_causal_lm(loss_value=0.0)  # exp(0) == 1.0

    with patch("rag_defenses.perplexity.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("rag_defenses.perplexity.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        det = PerplexityDetector(model_name="fake-model")
        det.fit(["clean text 1", "clean text 2"])

    assert det.clean_scores is not None
    assert list(det.clean_scores) == [1.0, 1.0]
    assert det.two_sided is True


def test_perplexity_scorer_returns_exp_of_loss_per_text():
    mock_tokenizer, mock_model = _mock_causal_lm(loss_value=1.0)  # exp(1) ~= 2.71828

    with patch("rag_defenses.perplexity.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("rag_defenses.perplexity.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        scorer = PerplexityScorer(model_name="fake-model", device="cpu")
        result = scorer.score_texts(["text a", "text b"])

    assert result.shape == (2,)
    assert result[0].item() == torch.tensor(1.0).exp().item()


def test_perplexity_detector_sets_pad_token_when_missing():
    mock_tokenizer, mock_model = _mock_causal_lm(loss_value=0.0)
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    with patch("rag_defenses.perplexity.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("rag_defenses.perplexity.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        PerplexityDetector(model_name="fake-model")

    assert mock_tokenizer.pad_token == "<eos>"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest defenses/tests/test_perplexity.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_defenses.perplexity'`

- [ ] **Step 3: Implement `perplexity.py`**

Create `defenses/src/rag_defenses/perplexity.py` (relocated from `RAG_Setting/src/medrag_repro/defense/perplexity.py`'s `PerplexityDetector`, with `BaseDetector` import repointed, plus `Agent_Setting/ReAct/defense_baselines.py`'s `PerplexityScorer`):

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_defenses.common import BaseDetector


class PerplexityDetector(BaseDetector):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        max_length: int = 512,
    ):
        super().__init__(two_sided=True, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.clean_scores: np.ndarray | None = None

    @torch.no_grad()
    def _perplexity(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss.item()
        return float(np.exp(loss))

    def fit(self, clean_texts: Sequence[str]) -> None:
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        scores = [self._perplexity(t) for t in texts]
        return np.array(scores, dtype=np.float64)


class PerplexityScorer:
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def score_texts(self, texts):
        scores = []
        for text in texts:
            tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            scores.append(torch.exp(outputs.loss).detach().cpu())
        return torch.stack(scores).float()
```

Note: `PerplexityScorer.__init__` calls `.to(device)` with `device="cuda"` by default (unchanged from the original — this default assumes a GPU is available; behavior-preserving relocation, not a place to "fix" for CPU-only environments).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest defenses/tests/test_perplexity.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add defenses/src/rag_defenses/perplexity.py defenses/tests/test_perplexity.py
git commit -m "feat(defenses): add rag_defenses.perplexity (PerplexityDetector + PerplexityScorer, kept separate)"
```

---

### Task 5: `rag_defenses.defense_baselines` (`QuantileStats`, `PerplexityStats`, `fit_upper_quantile`, `fit_two_sided_quantile`)

**Files:**
- Create: `defenses/src/rag_defenses/defense_baselines.py`
- Test: `defenses/tests/test_defense_baselines.py`

**Interfaces:**
- Consumes: nothing from Tasks 1-4 (independent module, no `BaseDetector` dependency — these are standalone threshold-fitting helpers, not `Detector` subclasses).
- Produces: `rag_defenses.defense_baselines.{QuantileStats, PerplexityStats, fit_upper_quantile, fit_two_sided_quantile}`.

- [ ] **Step 1: Write the failing tests**

Create `defenses/tests/test_defense_baselines.py`:

```python
from __future__ import annotations

import torch

from rag_defenses.defense_baselines import (
    PerplexityStats,
    QuantileStats,
    fit_two_sided_quantile,
    fit_upper_quantile,
)


def test_fit_upper_quantile_computes_threshold_and_false_positive_rate():
    clean_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    result = fit_upper_quantile(clean_scores, quantile=0.8)

    assert isinstance(result, QuantileStats)
    assert result.threshold == torch.quantile(clean_scores, 0.8).item()
    assert result.false_positive_rate == (clean_scores > result.threshold).float().mean().item()


def test_fit_two_sided_quantile_computes_symmetric_tail_thresholds():
    clean_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])

    result = fit_two_sided_quantile(clean_scores, quantile=0.8)

    assert isinstance(result, PerplexityStats)
    tail = (1 - 0.8) / 2
    assert result.lower_threshold == torch.quantile(clean_scores, tail).item()
    assert result.upper_threshold == torch.quantile(clean_scores, 1 - tail).item()


def test_fit_two_sided_quantile_clamps_tiny_tail_to_minimum():
    clean_scores = torch.tensor([1.0, 2.0, 3.0])

    result = fit_two_sided_quantile(clean_scores, quantile=0.9999)

    # tail = (1-0.9999)/2 = 0.00005, clamped to 1e-4
    expected_tail = 1e-4
    assert result.lower_threshold == torch.quantile(clean_scores, expected_tail).item()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest defenses/tests/test_defense_baselines.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_defenses.defense_baselines'`

- [ ] **Step 3: Implement `defense_baselines.py`**

Create `defenses/src/rag_defenses/defense_baselines.py` (relocated unchanged from `Agent_Setting/ReAct/defense_baselines.py`'s remaining dataclasses/functions — `l2_norm_scores` moved to Task 2, `l2_distance_scores`/`leave_one_out_l2_distance_scores` moved to Task 3, `PerplexityScorer` moved to Task 4):

```python
from dataclasses import dataclass

import torch


@dataclass
class QuantileStats:
    threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float


@dataclass
class PerplexityStats:
    lower_threshold: float
    upper_threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float


def fit_upper_quantile(clean_scores: torch.Tensor, quantile: float = 0.99) -> QuantileStats:
    threshold = torch.quantile(clean_scores, quantile).item()
    false_positive_rate = (clean_scores > threshold).float().mean().item()
    return QuantileStats(
        threshold=threshold,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
    )


def fit_two_sided_quantile(clean_scores: torch.Tensor, quantile: float = 0.99) -> PerplexityStats:
    tail = max((1 - quantile) / 2, 1e-4)
    lower = torch.quantile(clean_scores, tail).item()
    upper = torch.quantile(clean_scores, 1 - tail).item()
    false_positive_rate = ((clean_scores < lower) | (clean_scores > upper)).float().mean().item()
    return PerplexityStats(
        lower_threshold=lower,
        upper_threshold=upper,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest defenses/tests/test_defense_baselines.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add defenses/src/rag_defenses/defense_baselines.py defenses/tests/test_defense_baselines.py
git commit -m "feat(defenses): add rag_defenses.defense_baselines (QuantileStats/PerplexityStats fitting)"
```

---

### Task 6: RAG_Setting adapter — `defense/{common,l2_norm,l2_distance,perplexity}.py` become thin re-exports, `defense/drs.py`'s `BaseDetector` import repointed

**Files:**
- Modify: `RAG_Setting/src/medrag_repro/defense/common.py`
- Modify: `RAG_Setting/src/medrag_repro/defense/l2_norm.py`
- Modify: `RAG_Setting/src/medrag_repro/defense/l2_distance.py`
- Modify: `RAG_Setting/src/medrag_repro/defense/perplexity.py`
- Modify: `RAG_Setting/src/medrag_repro/defense/drs.py`
- Modify: `RAG_Setting/requirements.txt`
- Test: `RAG_Setting/tests/test_defenses_parity.py`

**Interfaces:**
- Consumes: `rag_defenses.{common.BaseDetector, l2_norm.L2NormDetector, l2_distance.L2DistanceDetector, perplexity.PerplexityDetector}` (Tasks 1-4).
- Produces: all four names remain importable from their existing `medrag_repro.defense.*` paths with identical behavior — used unchanged by `RAG_Setting/scripts/run_defense.py`'s `build_detector` function.

- [ ] **Step 1: Write the failing parity test**

Create `RAG_Setting/tests/test_defenses_parity.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest RAG_Setting/tests/test_defenses_parity.py -v`
Expected: FAIL — `assert False` (currently `medrag_repro`'s own local classes, not `rag_defenses`'s)

- [ ] **Step 3: Replace the four files with thin re-exports**

Replace the full contents of `RAG_Setting/src/medrag_repro/defense/common.py` with:

```python
from __future__ import annotations

from rag_defenses.common import BaseDetector

__all__ = ["BaseDetector"]
```

Replace the full contents of `RAG_Setting/src/medrag_repro/defense/l2_norm.py` with:

```python
from __future__ import annotations

from rag_defenses.l2_norm import L2NormDetector

__all__ = ["L2NormDetector"]
```

Replace the full contents of `RAG_Setting/src/medrag_repro/defense/l2_distance.py` with:

```python
from __future__ import annotations

from rag_defenses.l2_distance import L2DistanceDetector

__all__ = ["L2DistanceDetector"]
```

Replace the full contents of `RAG_Setting/src/medrag_repro/defense/perplexity.py` with:

```python
from __future__ import annotations

from rag_defenses.perplexity import PerplexityDetector

__all__ = ["PerplexityDetector"]
```

- [ ] **Step 4: Repoint `defense/drs.py`'s `BaseDetector` import**

In `RAG_Setting/src/medrag_repro/defense/drs.py`, change the line:

```python
from medrag_repro.defense.common import BaseDetector
```

to:

```python
from rag_defenses.common import BaseDetector
```

This is the ONLY change to this file — `DRSDetector`'s class body, `fit`/`score_texts` methods, and its delegation to `drs_defense.core` are untouched. Read the file first to confirm this is the only `medrag_repro.defense.common` reference in it.

- [ ] **Step 5: Add the `rag-defenses` dependency**

Modify `RAG_Setting/requirements.txt` (currently `-e .` / `-e ../drs_defense` / `-e ../infra` / `-e ../attacks` / `pytest`) to add the new editable dependency:

```
-e .
-e ../drs_defense
-e ../infra
-e ../attacks
-e ../defenses
pytest
```

- [ ] **Step 6: Run parity test to verify it passes**

Run: `pytest RAG_Setting/tests/test_defenses_parity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 7: Run the full RAG_Setting test suite to confirm nothing broke**

Run: `pytest RAG_Setting/tests/ -v`
Expected: all tests PASS, including the pre-existing DRS/LLM-client/L2-norm(Phase 5)/attack parity suites — the pre-existing `test_l2_norm_detector_parity.py` (Phase 5) should still pass since `L2NormDetector`'s behavior is unchanged, only its location moved twice (RAG_Setting → infra math only in Phase 5 → now the whole class into defenses/ in this phase).

- [ ] **Step 8: Commit**

```bash
git add RAG_Setting/src/medrag_repro/defense/common.py RAG_Setting/src/medrag_repro/defense/l2_norm.py RAG_Setting/src/medrag_repro/defense/l2_distance.py RAG_Setting/src/medrag_repro/defense/perplexity.py RAG_Setting/src/medrag_repro/defense/drs.py RAG_Setting/requirements.txt RAG_Setting/tests/test_defenses_parity.py
git commit -m "refactor(RAG_Setting): delegate defense detectors to rag_defenses, keep call signatures"
```

---

### Task 7: Agent_Setting adapter — `ReAct/defense_baselines.py` becomes a full re-export

**Files:**
- Modify: `Agent_Setting/ReAct/defense_baselines.py`
- Modify: `Agent_Setting/environment.yml`
- Test: `Agent_Setting/tests/test_defense_baselines_parity.py`

**Interfaces:**
- Consumes: `rag_defenses.l2_norm.l2_norm_scores`, `rag_defenses.l2_distance.{l2_distance_scores, leave_one_out_l2_distance_scores}`, `rag_defenses.perplexity.PerplexityScorer`, `rag_defenses.defense_baselines.{QuantileStats, PerplexityStats, fit_upper_quantile, fit_two_sided_quantile}` (Tasks 2-5).
- Produces: all eight names remain importable from `ReAct.defense_baselines` with identical signatures — used unchanged by `Agent_Setting/ReAct/local_wikienv.py`'s `from ReAct.defense_baselines import (...)` block.

- [ ] **Step 1: Write the failing parity test**

Create `Agent_Setting/tests/test_defense_baselines_parity.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Agent_Setting/tests/test_defense_baselines_parity.py -v`
Expected: FAIL — `assert False` (currently `ReAct`'s own local definitions, not `rag_defenses`'s)

- [ ] **Step 3: Replace the file with a full re-export**

Replace the full contents of `Agent_Setting/ReAct/defense_baselines.py` with:

```python
from rag_defenses.defense_baselines import (
    PerplexityStats,
    QuantileStats,
    fit_two_sided_quantile,
    fit_upper_quantile,
)
from rag_defenses.l2_distance import l2_distance_scores, leave_one_out_l2_distance_scores
from rag_defenses.l2_norm import l2_norm_scores
from rag_defenses.perplexity import PerplexityScorer

__all__ = [
    "PerplexityScorer",
    "PerplexityStats",
    "QuantileStats",
    "fit_two_sided_quantile",
    "fit_upper_quantile",
    "l2_distance_scores",
    "l2_norm_scores",
    "leave_one_out_l2_distance_scores",
]
```

- [ ] **Step 4: Add the `rag-defenses` dependency**

Modify `Agent_Setting/environment.yml`: in the `pip:` list, add `- -e ../defenses` next to the existing `- -e ../drs_defense` and `- -e ../infra` lines:

```yaml
    - -e ../drs_defense
    - -e ../infra
    - -e ../defenses
    - pytest
```

- [ ] **Step 5: Run parity test to verify it passes**

Run: `pytest Agent_Setting/tests/test_defense_baselines_parity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full Agent_Setting test suite to confirm nothing broke**

Run: `pytest Agent_Setting/tests/ -v`
Expected: all tests PASS, including the pre-existing DRS/ollama-client/L2-norm(Phase 5) parity suites — `test_l2_norm_scores_parity.py` (Phase 5) should still pass since `l2_norm_scores`'s behavior is unchanged.

- [ ] **Step 7: Commit**

```bash
git add Agent_Setting/ReAct/defense_baselines.py Agent_Setting/environment.yml Agent_Setting/tests/test_defense_baselines_parity.py
git commit -m "refactor(Agent_Setting): delegate defense_baselines to rag_defenses, keep call signatures"
```

---

### Task 8: Documentation — `rag_defenses` across the doc surfaces

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:** none — documentation only.

- [ ] **Step 1: Update `CLAUDE.md`**

Add a new bullet describing `defenses/` alongside the existing `drs_defense/`, `infra/`, `attacks/` bullets in the "Repository overview" section, following the same style: what it holds (`common.BaseDetector`, `l2_norm`, `l2_distance`, `perplexity`, `defense_baselines`), which subproject files are now thin adapters, the "do not reimplement locally; add it here and delegate" instruction, and a note that the two perplexity classes and two l2_distance formulas are kept separate (not merged) since they weren't proven interchangeable. Also add `defenses/tests/` to the Tests section's list of pytest suites.

- [ ] **Step 2: Update root `README.md`**

Add an equivalent one-line mention of `defenses/`/`rag_defenses`, matching the existing `attacks/`/`infra/`/`drs_defense/` bullets' level of detail.

- [ ] **Step 3: Verify scope**

Run: `git status --porcelain -- CLAUDE.md README.md`
Confirm only these two files show as modified for this task.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document rag_defenses across CLAUDE.md and README.md"
```

---

### Task 9: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none — this task only runs the test surfaces produced by Tasks 1-8.

- [ ] **Step 1: Run every test suite in the repo**

```bash
pytest defenses/tests/ -v
pytest attacks/tests/ -v
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest RAG_Setting/tests/ -v
pytest Agent_Setting/tests/ -v
pytest Retrieving_stage/tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Grep for any remaining direct definitions of the moved code**

```bash
grep -rn "^class BaseDetector\|^class L2NormDetector\|^class L2DistanceDetector\|^class PerplexityDetector\|^class PerplexityScorer" RAG_Setting/src Agent_Setting/ReAct
grep -rn "^def l2_distance_scores\|^def leave_one_out_l2_distance_scores\|^def fit_upper_quantile\|^def fit_two_sided_quantile" Agent_Setting/ReAct
```

Expected: no output — confirms the moved definitions no longer exist outside `defenses/src/rag_defenses/`.

- [ ] **Step 3: Verify `run_defense.py` and `local_wikienv.py` still import and construct their detectors successfully**

Run: `python3 -c "import sys; sys.path.insert(0, 'RAG_Setting/scripts'); import run_defense"` (confirm no `ImportError`/`ModuleNotFoundError` importing the script's module-level dependencies)

- [ ] **Step 4: Report results to the user**

Summarize: which files were created/modified, all test results, and confirm the four Detector classes plus the baseline-fitting utilities are now single-sourced in `rag_defenses`, with the two l2-distance formulas and two perplexity implementations correctly kept as distinct, non-merged names. No commit needed for this task (verification only).
