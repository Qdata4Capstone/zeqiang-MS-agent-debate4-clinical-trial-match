# defenses/l2_norm Extraction (Phase 5, narrowed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the one genuinely-duplicated defense scoring formula — L2-norm-of-embedding scoring, currently reimplemented once in numpy (`RAG_Setting/src/medrag_repro/defense/l2_norm.py`) and once in torch (`Agent_Setting/ReAct/defense_baselines.py`) — into a shared `rag_infra.defenses.l2_norm` module, following the exact `drs_defense`-adapter pattern already established in this repo (shared numpy math + per-subproject thin adapter that converts at the numpy/torch boundary + a parity test proving the adapter matches the shared function, not a reimplementation of it).

**Architecture:** This is a narrowed version of the design spec's Phase 5 (`defenses/`). Research before writing this plan found the phase's other named pieces don't consolidate cleanly:
- **L2-distance** is NOT the same formula in both places: `RAG_Setting`'s `L2DistanceDetector` scores distance-to-the-centroid-of-clean-embeddings; `Agent_Setting`'s `l2_distance_scores` scores distance-to-the-nearest-individual-clean-embedding (a KNN-style score, via `torch.cdist(...).min(dim=1)`). These are two different detection statistics that happen to share a name — like Phase 4's two "PoisonedRAG" attacks, there's no shared output to prove parity against, so nothing to extract-and-merge. Left as-is in both places.
- **Perplexity** (`RAG_Setting`'s `PerplexityDetector`, `Agent_Setting`'s `PerplexityScorer`) genuinely computes the same core value (`exp(causal-LM loss)`) in both places, but extracting it reopens the exact dependency question Phase 3 (`infra/retrieval/`) was deferred over: it would be the first `rag_infra` module needing `torch`+`transformers` as hard dependencies, plus a full causal-LM download/load at runtime. Deferred alongside Phase 3, not attempted here.
- **`BaseDetector`** (`RAG_Setting/src/medrag_repro/defense/common.py`) is a generic ABC with no domain coupling, but nothing outside `RAG_Setting` uses it (`Agent_Setting` has no equivalent base-class pattern) — there's nothing to deduplicate, so it stays `RAG_Setting`-local.

Unlike Phase 4 (nothing to extract at all), Phase 5 does have one real, provable duplicate: `np.linalg.norm(X, axis=1)` (RAG_Setting) and `torch.norm(embeddings.float(), dim=1)` (Agent_Setting) compute the identical value. That becomes `rag_infra.defenses.l2_norm.l2_norm_score(embeddings: np.ndarray) -> np.ndarray`, mirroring how `drs_defense.core` holds pure-numpy math while `Agent_Setting/ReAct/drs.py` wraps it at the torch boundary.

This plan also proactively updates the four documentation surfaces (`infra/README.md`, `infra/pyproject.toml`, `CLAUDE.md`, root `README.md`) in the same plan rather than waiting for the final whole-branch review to catch it — that gap has recurred in both Phase 1 and Phase 2's final reviews.

**Tech Stack:** Python, `numpy` (new hard dependency for `rag_infra` — not previously needed by `rag_infra.llm`/`rag_infra.data`), `torch` (already a dependency of `Agent_Setting`'s env, used only in that subproject's adapter, not added to `rag_infra` itself), `pytest`.

## Global Constraints

- `rag_infra` must stay importable under both Python 3.9 (`Agent_Setting`) and Python 3.10 (`RAG_Setting`) — no syntax newer than 3.9.
- `rag_infra.defenses.l2_norm` itself must be pure numpy — no `torch` import inside `infra/`. Torch conversion happens only in `Agent_Setting/ReAct/defense_baselines.py`'s adapter, which already has `torch` as an existing dependency.
- No changes to call-site behavior: `RAG_Setting/scripts/run_defense.py` constructs `L2NormDetector(encoder=..., lower_quantile=..., upper_quantile=...)` unchanged; `Agent_Setting/ReAct/local_wikienv.py` imports and calls `l2_norm_scores(...)` unchanged from `ReAct.defense_baselines`. Neither caller file is touched by this plan.
- `Agent_Setting/ReAct/defense_baselines.py`'s other functions (`l2_distance_scores`, `leave_one_out_l2_distance_scores`, `fit_upper_quantile`, `fit_two_sided_quantile`, `PerplexityScorer`) and `RAG_Setting`'s other detector files (`l2_distance.py`, `perplexity.py`, `common.py`) are NOT touched by this plan — only `l2_norm.py` (`RAG_Setting`) and the `l2_norm_scores` function inside `defense_baselines.py` (`Agent_Setting`) change.
- `l2_norm_scores`'s existing 1D-input handling (`if embeddings.dim() == 1: embeddings = embeddings.unsqueeze(0)`) must be preserved exactly — it's used both on batches (`self.clean_reference_embeddings`) and single embeddings (`poison_embedding`, where the caller then calls `.item()` on the result).
- Follow the `drs_defense`/`ReAct/drs.py` precedent exactly: shared pure-numpy math + adapter that detaches/converts at the numpy↔torch boundary (mirroring `ReAct/drs.py`'s existing `clean_embeddings.detach().cpu().double().numpy()` pattern) + a parity test proving the adapter's output matches the shared function's output on the same input (not an identity/`is` check, since real conversion logic sits in between — mirroring `test_drs_detector_matches_drs_defense_core`'s and `test_fit_and_score_drs_match_drs_defense_core`'s existing style, not Phase 1/2's `is`-based style).
- Run all commands from the `safematch_v3` worktree root: `/Users/qiyanjun/Code/Public/zeqiang-MS-agent-debate4-clinical-trial-match/.worktrees/safematch_v3`. `rag_infra` is already installed editable in this environment from Phases 1-2.

---

### Task 1: `rag_infra.defenses.l2_norm` (shared L2-norm scoring) + numpy dependency

**Files:**
- Modify: `infra/pyproject.toml`
- Create: `infra/src/rag_infra/defenses/__init__.py`
- Create: `infra/src/rag_infra/defenses/l2_norm.py`
- Test: `infra/tests/test_defenses_l2_norm.py`

**Interfaces:**
- Produces: `rag_infra.defenses.l2_norm.l2_norm_score(embeddings: np.ndarray) -> np.ndarray`

- [ ] **Step 1: Add `numpy` as a dependency**

In `infra/pyproject.toml`, the `dependencies` list currently reads:

```toml
dependencies = [
  "openai>=1.30.0",
  "requests>=2.31",
]
```

Add `"numpy>=1.24"` (matching the version floor already used by `drs_defense/pyproject.toml`), so the block reads:

```toml
dependencies = [
  "openai>=1.30.0",
  "requests>=2.31",
  "numpy>=1.24",
]
```

- [ ] **Step 2: Create the `defenses` subpackage**

Create `infra/src/rag_infra/defenses/__init__.py` (empty file).

- [ ] **Step 3: Write the failing tests**

Create `infra/tests/test_defenses_l2_norm.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest infra/tests/test_defenses_l2_norm.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_infra.defenses'`

- [ ] **Step 5: Implement `l2_norm.py`**

Create `infra/src/rag_infra/defenses/l2_norm.py`:

```python
from __future__ import annotations

import numpy as np


def l2_norm_score(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]
    return np.linalg.norm(embeddings, axis=1)
```

This is the shared core of both `RAG_Setting`'s `L2NormDetector.score_texts` (`np.linalg.norm(X, axis=1)`, always called on a 2D batch from an encoder) and `Agent_Setting`'s `l2_norm_scores` (`torch.norm(embeddings.float(), dim=1)`, called on both 2D batches and single 1D embeddings — hence the 1D-handling branch, which mirrors `l2_norm_scores`'s existing `if embeddings.dim() == 1: embeddings = embeddings.unsqueeze(0)`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest infra/tests/test_defenses_l2_norm.py -v`
Expected: PASS (2 passed)

- [ ] **Step 7: Reinstall to pick up the new dependency, then run the full infra test suite**

Run: `pip install -e infra` (picks up the new `numpy` dependency declaration; harmless no-op if numpy is already present in this environment)
Run: `pytest infra/tests/ -v`
Expected: all PASS (existing `llm`/`data` tests plus the 2 new ones)

- [ ] **Step 8: Commit**

```bash
git add infra/pyproject.toml infra/src/rag_infra/defenses/__init__.py infra/src/rag_infra/defenses/l2_norm.py infra/tests/test_defenses_l2_norm.py
git commit -m "feat(infra): add rag_infra.defenses.l2_norm (shared L2-norm scoring math)"
```

---

### Task 2: RAG_Setting adapter — `L2NormDetector` delegates to `rag_infra.defenses.l2_norm`

**Files:**
- Modify: `RAG_Setting/src/medrag_repro/defense/l2_norm.py`
- Test: `RAG_Setting/tests/test_l2_norm_detector_parity.py`

**Interfaces:**
- Consumes: `rag_infra.defenses.l2_norm.l2_norm_score` (Task 1).
- Produces: `L2NormDetector`'s public interface (constructor signature, `fit`, `score_texts`, `detect` — all inherited from `BaseDetector` except `fit`/`score_texts`) is completely unchanged; only `score_texts`'s internal implementation delegates to the shared function. `RAG_Setting/scripts/run_defense.py`'s `L2NormDetector(encoder=..., lower_quantile=..., upper_quantile=...)` construction is untouched by this task.

- [ ] **Step 1: Write the failing parity test**

Create `RAG_Setting/tests/test_l2_norm_detector_parity.py`:

```python
from __future__ import annotations

import numpy as np

from medrag_repro.defense.l2_norm import L2NormDetector
from rag_infra.defenses.l2_norm import l2_norm_score


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
```

(The `_FakeEncoder` fixture is copied from the existing `RAG_Setting/tests/test_drs_detector_parity.py` — that file doesn't export it for reuse, and there's no shared `conftest.py` in `RAG_Setting/tests/`, so redefine it locally in this new test file rather than importing across test modules.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest RAG_Setting/tests/test_l2_norm_detector_parity.py -v`
Expected: FAIL — the assertion currently passes trivially only because `L2NormDetector.score_texts` already computes the same numpy formula inline; to make this a meaningful RED step, temporarily verify failure by checking the test imports `rag_infra.defenses.l2_norm` (which doesn't exist yet at this point in the task sequence — Task 1 must be complete first, which it is, since tasks run in order) — actually confirm failure the standard way: since `rag_infra.defenses.l2_norm` DOES exist by this point (Task 1 already committed it) and `L2NormDetector.score_texts` hasn't been changed yet, this test should currently PASS already (both sides compute the identical numpy formula independently). This is expected and fine — proceed to Step 3 regardless of RED/GREEN here, since the point of this task is the *delegation* (removing the duplicate formula), not changing the output. Skip re-verifying failure; note in your report that this parity test passes both before and after the refactor, and that the task's real verification is the code-reading confirmation in Step 4.

- [ ] **Step 3: Change `score_texts` to delegate to the shared function**

In `RAG_Setting/src/medrag_repro/defense/l2_norm.py`, add an import and change the `score_texts` method body:

```python
from __future__ import annotations

from typing import Sequence

import numpy as np

from medrag_repro.defense.common import BaseDetector
from medrag_repro.retriever.contriever import ContrieverEncoder
from rag_infra.defenses.l2_norm import l2_norm_score


class L2NormDetector(BaseDetector):
    def __init__(self, encoder: ContrieverEncoder, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
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
```

(Only the import block and `score_texts`'s return line change — `__init__` and `fit` are untouched.)

- [ ] **Step 4: Run the parity test and confirm delegation by reading the diff**

Run: `pytest RAG_Setting/tests/test_l2_norm_detector_parity.py -v`
Expected: PASS (1 passed)

Then confirm via `git diff` that `score_texts` now calls `l2_norm_score(X)` rather than computing `np.linalg.norm(X, axis=1)` inline — this is the actual evidence of delegation, since the parity test alone can't distinguish "delegates to the shared function" from "still computes the same formula independently."

- [ ] **Step 5: Run the full RAG_Setting test suite to confirm nothing broke**

Run: `pytest RAG_Setting/tests/ -v`
Expected: all tests PASS, including the pre-existing `test_drs_detector_parity.py` and `test_llm_client_parity.py`

- [ ] **Step 6: Commit**

```bash
git add RAG_Setting/src/medrag_repro/defense/l2_norm.py RAG_Setting/tests/test_l2_norm_detector_parity.py
git commit -m "refactor(RAG_Setting): delegate L2NormDetector scoring to rag_infra.defenses.l2_norm"
```

---

### Task 3: Agent_Setting adapter — `l2_norm_scores` delegates to `rag_infra.defenses.l2_norm`

**Files:**
- Modify: `Agent_Setting/ReAct/defense_baselines.py`
- Test: `Agent_Setting/tests/test_l2_norm_scores_parity.py`

**Interfaces:**
- Consumes: `rag_infra.defenses.l2_norm.l2_norm_score` (Task 1).
- Produces: `l2_norm_scores(embeddings: torch.Tensor) -> torch.Tensor`'s public signature and behavior (including 1D-input handling) are unchanged; `Agent_Setting/ReAct/local_wikienv.py`'s calls (`l2_norm_scores(self.clean_reference_embeddings)`, `l2_norm_scores(poison_embedding)`) are untouched by this task. The other five names in `defense_baselines.py` (`l2_distance_scores`, `leave_one_out_l2_distance_scores`, `fit_upper_quantile`, `fit_two_sided_quantile`, `PerplexityScorer`) are untouched.

- [ ] **Step 1: Write the failing parity test**

Create `Agent_Setting/tests/test_l2_norm_scores_parity.py`:

```python
from __future__ import annotations

import numpy as np
import pytest
import torch

from rag_infra.defenses.l2_norm import l2_norm_score as core_l2_norm_score
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Agent_Setting/tests/test_l2_norm_scores_parity.py -v`
Expected: PASS already (both sides currently compute the identical formula independently — same situation as Task 2 Step 2; this is fine, the real verification is the delegation check in Step 4)

- [ ] **Step 3: Change `l2_norm_scores` to delegate to the shared function**

In `Agent_Setting/ReAct/defense_baselines.py`, add an import at the top and replace the function body:

```python
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_infra.defenses.l2_norm import l2_norm_score as _l2_norm_score_np


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


def l2_norm_scores(embeddings: torch.Tensor) -> torch.Tensor:
    device = embeddings.device
    scores_np = _l2_norm_score_np(embeddings.detach().cpu().float().numpy())
    return torch.from_numpy(scores_np).float().to(device)
```

(Everything from `l2_distance_scores` onward — the rest of the original file — is unchanged; only the top import block and the `l2_norm_scores` function body change. This mirrors `Agent_Setting/ReAct/drs.py`'s existing `.detach().cpu().double().numpy()` conversion pattern, using `.float()` instead of `.double()` to match `l2_norm_scores`'s original `float32`-preserving behavior.)

- [ ] **Step 4: Run the parity test and confirm delegation by reading the diff**

Run: `pytest Agent_Setting/tests/test_l2_norm_scores_parity.py -v`
Expected: PASS (3 passed)

Then confirm via `git diff` that `l2_norm_scores` now calls `_l2_norm_score_np(...)` rather than computing `torch.norm(embeddings.float(), dim=1)` inline, and that `l2_distance_scores` and everything below it in the file is untouched.

- [ ] **Step 5: Run the full Agent_Setting test suite to confirm nothing broke**

Run: `pytest Agent_Setting/tests/ -v`
Expected: all tests PASS, including the pre-existing `test_drs_parity.py` and `test_ollama_client_parity.py`

- [ ] **Step 6: Commit**

```bash
git add Agent_Setting/ReAct/defense_baselines.py Agent_Setting/tests/test_l2_norm_scores_parity.py
git commit -m "refactor(Agent_Setting): delegate l2_norm_scores to rag_infra.defenses.l2_norm"
```

---

### Task 4: Documentation — `rag_infra.defenses` across the four doc surfaces

**Files:**
- Modify: `infra/README.md`
- Modify: `infra/pyproject.toml`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Interfaces:** none — documentation only, no code interfaces.

- [ ] **Step 1: Update `infra/README.md`**

Read the file first (it currently has `## rag_infra.llm` and `## rag_infra.data` sections, added in Phases 1-2). Add a `## rag_infra.defenses` section in the same style, documenting `l2_norm.py`'s `l2_norm_score` function and its two adapters (`RAG_Setting`'s `L2NormDetector`, `Agent_Setting`'s `l2_norm_scores`). Also add a short deferral note (mirroring the existing `generate_with_ollama`/`medrag_repro/utils/io.py` deferral notes already in this file) recording that:
- `l2_distance` was evaluated but NOT extracted — `RAG_Setting`'s centroid-distance and `Agent_Setting`'s nearest-neighbor-distance are different formulas, not duplicates.
- `perplexity` was evaluated but NOT extracted — same underlying computation (`exp(causal-LM loss)`) in both places, but deferred alongside `infra/retrieval/` (Phase 3) pending a decision on adding `torch`+`transformers` as hard dependencies of `rag_infra`.

- [ ] **Step 2: Update `infra/pyproject.toml`'s `description`**

Broaden the `description` field (already covers LLM-client and dataset file-I/O infrastructure from Phases 1-2) to also mention defense-scoring helpers, e.g. append "and defense-scoring math (L2-norm)".

- [ ] **Step 3: Update `CLAUDE.md`**

Find the `infra/` bullet in the "Repository overview" section (already mentions `rag_infra.llm` and `rag_infra.data.jsonl` from the Phase 1/2 doc fixes) and extend it with `rag_infra.defenses.l2_norm`, naming its two adapters, with the same "do not reimplement locally; add it here and delegate" instruction already used for the other two subpackages. Also extend the Tests section line to mention `rag_infra.defenses`.

- [ ] **Step 4: Update root `README.md`**

Read the file first (it has an `infra/` bullet from the Phase 1/2 doc fixes covering `rag_infra.llm` and `rag_infra.data.jsonl`). Add an equivalent one-line mention of `rag_infra.defenses.l2_norm` if the README documents things at that level of detail — match whatever level of detail the existing `infra/` bullet already uses.

- [ ] **Step 5: Verify nothing else changed**

Run: `git status --porcelain -- infra/README.md infra/pyproject.toml CLAUDE.md README.md`
Confirm only these four files show as modified, and `git diff --stat` shows no changes outside these four files for this task.

- [ ] **Step 6: Commit**

```bash
git add infra/README.md infra/pyproject.toml CLAUDE.md README.md
git commit -m "docs: document rag_infra.defenses.l2_norm across the four doc surfaces"
```

---

### Task 5: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none — this task only runs the test surfaces produced by Tasks 1–4.

- [ ] **Step 1: Run every test suite in the repo**

```bash
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest RAG_Setting/tests/ -v
pytest Agent_Setting/tests/ -v
pytest Retrieving_stage/tests/ -v
```

Expected: all PASS. This confirms Phase 5's narrowed extraction didn't regress any prior phase's work.

- [ ] **Step 2: Grep for any remaining duplicate L2-norm formula definitions**

```bash
grep -rn "np.linalg.norm.*axis=1\|torch.norm.*dim=1" RAG_Setting/src/medrag_repro/defense/l2_norm.py Agent_Setting/ReAct/defense_baselines.py
```

Expected: no output — confirms neither file computes the L2-norm formula inline anymore (both now call `l2_norm_score`/`_l2_norm_score_np`).

- [ ] **Step 3: Report results to the user**

Summarize: which files were created/modified, all test results, and confirm `l2_norm_score` is now single-sourced in `rag_infra.defenses.l2_norm`. Note explicitly that `l2_distance` (different formulas) and `perplexity` (deferred pending the same dependency-footprint decision as Phase 3) remain subproject-local by design, not because they were missed — and that `BaseDetector` stays `RAG_Setting`-local since nothing else uses it. No commit needed for this task (verification only).
