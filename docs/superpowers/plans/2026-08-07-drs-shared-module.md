# Shared `drs_defense` Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate the repo's four independent, drifted reimplementations of the DRS (Directional Relative Shifts) poisoning defense into one paper-faithful, pip-installable `drs_defense` module, with a test suite that verifies correctness against the paper's Algorithm 1 / Eq. 3 and its central qualitative claims — then make each existing call site a thin adapter over it.

**Architecture:** A new top-level package `drs_defense/` (its own `pyproject.toml`, `src/drs_defense/` layout, pure-NumPy, no torch/GPU dependency) holds the single reference implementation (`core.py`) plus its own test suite. `Agent_Setting/ReAct/drs.py`, `RAG_Setting/src/medrag_repro/defense/drs.py`, `RAG_Setting/src/medrag_repro/defense/drs_old.py` (retired), and `Retrieving_stage/poisonrag_experiment/drs.py` are rewritten to delegate to `drs_defense.core` while preserving their exact existing call signatures, so none of their callers (`local_wikienv.py`, `run_defense.py`, `run_drs.py`, `run_poisonrag_experiment.py`) need to change. Each subproject adds `drs_defense` as a local editable dependency.

**Tech Stack:** Python, NumPy (core library), PyTorch (Agent_Setting adapter boundary only), pytest (new — no test infra exists anywhere in this repo today).

## Global Constraints

- `drs_defense`'s runtime dependency is NumPy only (`numpy>=1.24`), so it installs cleanly into all three existing environments (Agent_Setting: conda, Python 3.9; RAG_Setting: Python 3.10; Retrieving_stage: unpinned Python). `requires-python = ">=3.9"`.
- The paper-faithful formula, already fixed earlier in this session, must not regress: `DRS(z; X) = Σ_{i=1}^{M} |z^T v_i| / sqrt(λ_i)` over the `M` **smallest**-eigenvalue directions (ascending order), threshold = q-th quantile of clean scores. Source: "Understanding Data Poisoning Attacks for RAG: Insights and Algorithms" (ICLR 2025 submission), Algorithm 1 & Eq. 3, https://openreview.net/pdf?id=2aL6gcFX7q.
- Every adapter refactor must be behavior-preserving for its existing callers — verified by a parity test comparing the adapter's output to `drs_defense.core` directly on identical synthetic input. No changes to `local_wikienv.py`, `run_defense.py`, or `run_poisonrag_experiment.py` call sites.
- Editable installs use the relative path `../drs_defense`, run from inside each subproject's own directory — consistent with this repo's existing convention (e.g. `RAG_Setting/requirements.txt` already contains `-e .`, resolved relative to cwd).
- No placeholders, no TODOs — every step below contains the complete file content or exact diff to apply.

---

## Prior investigation (context for the implementer)

There are **four**, not three, independent DRS implementations in this repo, all derived from the same paper but hand-copied and diverged:

1. `Agent_Setting/ReAct/drs.py` — PyTorch, dataclass `DRSStats` + functions `fit_drs`/`score_drs`. Used by `Agent_Setting/ReAct/local_wikienv.py` (only reads `.threshold`, `.false_positive_rate`, `.num_directions` externally — confirmed by grep).
2. `RAG_Setting/src/medrag_repro/defense/drs.py` — NumPy, class `DRSDetector(BaseDetector)`. Used by `RAG_Setting/scripts/run_defense.py`.
3. `RAG_Setting/src/medrag_repro/defense/drs_old.py` — NumPy, function `compute_drs_reference` returning a dict + closure. Still actively used by `RAG_Setting/scripts/run_drs.py` (a separate, redundant pipeline step). Has the same `/λ` (not `/√λ`) bug that was just fixed in `drs.py`.
4. `Retrieving_stage/poisonrag_experiment/drs.py` — NumPy, functions `fit_drs`/`drs_score`/`drs_threshold`, dict-based model, with a local `power` ablation knob (CLI flag `--drs_power`, already fixed this session to default to `1.0` = the paper's formula). Used by `Retrieving_stage/poisonrag_experiment/run_poisonrag_experiment.py`.

This plan builds one reference implementation and turns all four into thin, tested adapters (with #3 retired outright since #2 already covers its use case).

---

## Task 1: Scaffold `drs_defense` and its standardization/eigenbasis primitives

**Files:**
- Create: `drs_defense/pyproject.toml`
- Create: `drs_defense/src/drs_defense/__init__.py`
- Create: `drs_defense/src/drs_defense/core.py`
- Test: `drs_defense/tests/test_core_eigenbasis.py`

**Interfaces:**
- Produces: `drs_defense.core.standardize(embeddings, eps=1e-8) -> (standardized, mean, std)`; `drs_defense.core.low_variance_eigenbasis(standardized, num_directions) -> (eigenvalues, eigenvectors)`.

- [ ] **Step 1: Create the package scaffold**

`drs_defense/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "drs-defense"
version = "0.1.0"
description = "Reference implementation of the DRS (Directional Relative Shifts) poisoning defense from 'Understanding Data Poisoning Attacks for RAG: Insights and Algorithms' (ICLR 2025 submission, openreview.net/pdf?id=2aL6gcFX7q)"
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "numpy>=1.24",
]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

`drs_defense/src/drs_defense/__init__.py` (leave empty for this step; populated in Task 3):
```python
```

- [ ] **Step 2: Write the failing test for the standardization/eigenbasis primitives**

`drs_defense/tests/test_core_eigenbasis.py`:
```python
import numpy as np
import pytest

from drs_defense.core import low_variance_eigenbasis, standardize


def test_standardize_zero_mean_unit_population_variance():
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, mean, std = standardize(X)

    np.testing.assert_allclose(standardized.mean(axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(standardized.std(axis=0), [1.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(mean, [[0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(std, [[np.sqrt(2.5), np.sqrt(2.5)]], atol=1e-12)


def test_low_variance_eigenbasis_selects_smallest_eigenvalue_direction():
    # Constructed so the standardized covariance (np.cov default ddof=1, N=4)
    # is exactly (4/3) * [[1, 0.6], [0.6, 1]], with eigenvalues 32/15 along
    # (1, 1) (high-variance) and 8/15 along (1, -1) (low-variance).
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, _, _ = standardize(X)

    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=1)

    assert eigenvalues.shape == (1,)
    assert eigenvectors.shape == (2, 1)
    np.testing.assert_allclose(eigenvalues[0], 8.0 / 15.0, rtol=1e-8)

    v = eigenvectors[:, 0]
    # low-variance eigenvector must point along (1, -1): equal magnitude, opposite sign
    np.testing.assert_allclose(abs(v[0]), abs(v[1]), rtol=1e-8)
    assert v[0] * v[1] < 0


def test_low_variance_eigenbasis_clips_num_directions_to_dimensionality():
    X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])
    standardized, _, _ = standardize(X)
    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions=1000)
    assert eigenvalues.shape == (2,)
    assert eigenvectors.shape == (2, 2)


def test_low_variance_eigenbasis_requires_at_least_two_samples():
    with pytest.raises(ValueError):
        low_variance_eigenbasis(np.zeros((1, 3)), num_directions=1)
```

- [ ] **Step 3: Run it to verify it fails**

```bash
cd drs_defense
pip install -e ".[dev]"
pytest tests/test_core_eigenbasis.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'drs_defense.core'`.

- [ ] **Step 4: Implement `standardize` and `low_variance_eigenbasis`**

`drs_defense/src/drs_defense/core.py`:
```python
"""Reference implementation of the DRS (Directional Relative Shifts) defense.

Source: "Understanding Data Poisoning Attacks for RAG: Insights and Algorithms"
(ICLR 2025 submission), https://openreview.net/pdf?id=2aL6gcFX7q, Section 4,
Algorithm 1 (Compute Directional Relevance Score) and Algorithm 2 (Detection
with DRS).

Algorithm 1 (Eq. 3):
    Given a standardized clean data matrix X in R^(n x d):
    1. Eigendecompose the covariance S = V * Lambda * V^T
    2. Sort eigenvalues (and eigenvectors) ascending: lambda_sigma(1) <= ... <= lambda_sigma(d)
    3. DRS(z; X) = sum_{i=1}^{M} |z^T v_sigma(i)| / sqrt(lambda_sigma(i))
       over the M *smallest*-eigenvalue ("low-variance") directions.

Algorithm 2:
    Fit DRS on top-K retrieved clean documents per protected query, set the
    decision threshold tau to the q-th quantile of the clean DRS scores, and
    reject any future document z with DRS(z; X_clean) > tau.
"""
from __future__ import annotations

import numpy as np

DEFAULT_EPS = 1e-8


def standardize(embeddings, eps: float = DEFAULT_EPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Column-wise zero-mean, unit-variance standardization.

    Returns (standardized, mean, std); mean/std are shaped (1, d) so they
    broadcast against any (n, d) batch.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D (n, d), got shape {X.shape}")
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    return (X - mean) / std, mean, std


def low_variance_eigenbasis(standardized: np.ndarray, num_directions: int) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1, steps 1-2: eigendecompose the covariance of standardized
    data and keep the `num_directions` eigenvectors with the *smallest*
    eigenvalues.

    Returns (eigenvalues, eigenvectors), ascending, truncated to
    min(num_directions, d) columns.
    """
    if standardized.shape[0] < 2:
        raise ValueError("At least 2 samples are required to estimate a covariance matrix.")
    cov = np.cov(standardized, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    m = min(num_directions, eigenvectors.shape[1])
    return eigenvalues[:m], eigenvectors[:, :m]
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_core_eigenbasis.py -v
```
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
git add drs_defense/pyproject.toml drs_defense/src/drs_defense/__init__.py \
        drs_defense/src/drs_defense/core.py drs_defense/tests/test_core_eigenbasis.py
git commit -m "feat(drs_defense): scaffold package and standardization/eigenbasis primitives"
```

---

## Task 2: `drs_score` (Eq. 3)

**Files:**
- Modify: `drs_defense/src/drs_defense/core.py`
- Test: `drs_defense/tests/test_core_score.py`

**Interfaces:**
- Consumes: `standardize`, `low_variance_eigenbasis` (Task 1).
- Produces: `drs_defense.core.DRSModel` (frozen dataclass: `mean`, `std`, `eigenvalues`, `eigenvectors`, `num_directions`, `eps`); `drs_defense.core.fit_drs(clean_embeddings, num_directions=100, eps=1e-8) -> DRSModel`; `drs_defense.core.drs_score(embeddings, model) -> float | np.ndarray` (scalar for a single `(d,)` input, `(n,)` array for a batch).

- [ ] **Step 1: Write the failing test**

`drs_defense/tests/test_core_score.py`:
```python
import numpy as np
import pytest

from drs_defense.core import drs_score, fit_drs

CLEAN_X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])


def test_drs_score_matches_hand_computed_value():
    # With num_directions=1 (keeping only the low-variance (1,-1) direction,
    # eigenvalue 8/15 -- see Task 1's derivation), the point z=(1,-1) has
    # DRS(z) = sqrt(3/2) exactly:
    #   z_std = (1,-1)/sqrt(2.5); |proj onto (1,-1)/sqrt(2)| = 2/sqrt(5)
    #   DRS = (2/sqrt(5)) / sqrt(8/15) = sqrt(3/2)
    model = fit_drs(CLEAN_X, num_directions=1)
    score = drs_score(np.array([1.0, -1.0]), model)
    assert score == pytest.approx(np.sqrt(1.5), rel=1e-8)


def test_drs_score_batch_matches_single():
    model = fit_drs(CLEAN_X, num_directions=2)
    points = np.array([[1.0, -1.0], [0.5, 0.5], [3.0, 0.0]])
    batch_scores = drs_score(points, model)
    single_scores = np.array([drs_score(p, model) for p in points])
    np.testing.assert_allclose(batch_scores, single_scores, rtol=1e-12)


def test_drs_score_of_mean_is_near_zero():
    model = fit_drs(CLEAN_X, num_directions=2)
    score = drs_score(np.array([0.0, 0.0]), model)
    assert score == pytest.approx(0.0, abs=1e-10)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pytest tests/test_core_score.py -v
```
Expected: FAIL — `ImportError: cannot import name 'fit_drs'`.

- [ ] **Step 3: Implement `DRSModel`, `fit_drs`, `drs_score`**

Append to `drs_defense/src/drs_defense/core.py`:
```python
from dataclasses import dataclass


@dataclass(frozen=True)
class DRSModel:
    """Fitted DRS reference model: mean/std/eigenbasis of the clean data."""

    mean: np.ndarray
    std: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    num_directions: int
    eps: float = DEFAULT_EPS


def fit_drs(clean_embeddings, num_directions: int = 100, eps: float = DEFAULT_EPS) -> DRSModel:
    """Fit a DRS model on clean embeddings (Algorithm 1, steps 1-2)."""
    standardized, mean, std = standardize(clean_embeddings, eps=eps)
    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions)
    return DRSModel(
        mean=mean,
        std=std,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        num_directions=eigenvectors.shape[1],
        eps=eps,
    )


def drs_score(embeddings, model: DRSModel):
    """Eq. 3: DRS(z; X) = sum_i |z^T v_i| / sqrt(lambda_i).

    Accepts a single embedding (d,) or a batch (n, d); returns a scalar
    (np.float64) for the single case, or a (n,) array for the batch case.
    """
    Z = np.asarray(embeddings, dtype=np.float64)
    single = Z.ndim == 1
    if single:
        Z = Z[None, :]
    standardized = (Z - model.mean) / model.std
    projections = np.abs(standardized @ model.eigenvectors)
    scales = np.sqrt(np.maximum(model.eigenvalues, model.eps))
    scores = (projections / scales).sum(axis=1)
    return scores[0] if single else scores
```

Move the `from dataclasses import dataclass` import to the top of the file with the other imports (next to `import numpy as np`) rather than leaving it inline — the step above shows it separately only to make the diff clear.

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_core_score.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add drs_defense/src/drs_defense/core.py drs_defense/tests/test_core_score.py
git commit -m "feat(drs_defense): implement DRSModel, fit_drs, and drs_score (Eq. 3)"
```

---

## Task 3: Threshold derivation (Algorithm 2) and public API surface

**Files:**
- Modify: `drs_defense/src/drs_defense/core.py`
- Modify: `drs_defense/src/drs_defense/__init__.py`
- Test: `drs_defense/tests/test_threshold.py`

**Interfaces:**
- Consumes: `fit_drs`, `drs_score` (Task 2).
- Produces: `drs_defense.core.quantile_threshold(scores, quantile=0.99) -> float`; `drs_defense.core.fit_drs_with_threshold(clean_embeddings, num_directions=100, quantile=0.99, eps=1e-8) -> (DRSModel, clean_scores, threshold)`; `drs_defense.core.is_flagged(scores, threshold) -> np.ndarray[bool]`.

- [ ] **Step 1: Write the failing test**

`drs_defense/tests/test_threshold.py`:
```python
import numpy as np
import pytest

from drs_defense.core import (
    drs_score,
    fit_drs,
    fit_drs_with_threshold,
    is_flagged,
    quantile_threshold,
)


def test_quantile_threshold_basic():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert quantile_threshold(scores, quantile=0.5) == pytest.approx(3.0)


def test_fit_drs_with_threshold_reproduces_manual_pipeline():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(200, 5))

    model, clean_scores, threshold = fit_drs_with_threshold(clean, num_directions=3, quantile=0.9)

    manual_model = fit_drs(clean, num_directions=3)
    np.testing.assert_allclose(model.eigenvalues, manual_model.eigenvalues)
    assert threshold == quantile_threshold(clean_scores, 0.9)


def test_fit_drs_with_threshold_empirical_fpr_on_held_out_clean_data():
    # Algorithm 2 calibration property: the fraction of *held-out* clean
    # samples exceeding tau should track (1 - quantile).
    rng = np.random.default_rng(42)
    fit_data = rng.normal(size=(1000, 5))
    holdout_data = rng.normal(size=(1000, 5))

    model, _, threshold = fit_drs_with_threshold(fit_data, num_directions=3, quantile=0.9)
    holdout_scores = drs_score(holdout_data, model)

    observed_fpr = is_flagged(holdout_scores, threshold).mean()
    assert abs(observed_fpr - 0.1) < 0.05
```

- [ ] **Step 2: Run it to verify it fails**

```bash
pytest tests/test_threshold.py -v
```
Expected: FAIL — `ImportError: cannot import name 'fit_drs_with_threshold'`.

- [ ] **Step 3: Implement the threshold functions**

Append to `drs_defense/src/drs_defense/core.py`:
```python
def quantile_threshold(scores, quantile: float = 0.99) -> float:
    """Algorithm 2, step 4: tau = q-th quantile of clean DRS scores."""
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))


def fit_drs_with_threshold(
    clean_embeddings,
    num_directions: int = 100,
    quantile: float = 0.99,
    eps: float = DEFAULT_EPS,
) -> tuple[DRSModel, np.ndarray, float]:
    """Algorithm 2, steps 1-4: fit + score clean data + derive tau."""
    model = fit_drs(clean_embeddings, num_directions=num_directions, eps=eps)
    clean_scores = drs_score(clean_embeddings, model)
    threshold = quantile_threshold(clean_scores, quantile)
    return model, clean_scores, threshold


def is_flagged(scores, threshold: float) -> np.ndarray:
    """Algorithm 2, step 5: reject z if DRS(z; X_clean) > tau."""
    return np.asarray(scores) > threshold
```

Populate `drs_defense/src/drs_defense/__init__.py`:
```python
from drs_defense.core import (
    DEFAULT_EPS,
    DRSModel,
    drs_score,
    fit_drs,
    fit_drs_with_threshold,
    is_flagged,
    low_variance_eigenbasis,
    quantile_threshold,
    standardize,
)

__all__ = [
    "DEFAULT_EPS",
    "DRSModel",
    "drs_score",
    "fit_drs",
    "fit_drs_with_threshold",
    "is_flagged",
    "low_variance_eigenbasis",
    "quantile_threshold",
    "standardize",
]
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
pytest tests/test_threshold.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add drs_defense/src/drs_defense/core.py drs_defense/src/drs_defense/__init__.py \
        drs_defense/tests/test_threshold.py
git commit -m "feat(drs_defense): add Algorithm 2 threshold helpers and public API"
```

---

## Task 4: Paper-fidelity verification tests

**Files:**
- Test: `drs_defense/tests/test_paper_claims.py`

**Interfaces:**
- Consumes: `fit_drs`, `drs_score`, `fit_drs_with_threshold`, `is_flagged` (Tasks 2-3). No production code changes in this task — it is purely a verification suite.

- [ ] **Step 1: Write the paper-claim tests**

`drs_defense/tests/test_paper_claims.py`:
```python
import numpy as np
import pytest

from drs_defense.core import drs_score, fit_drs, fit_drs_with_threshold, is_flagged

# Same 2x2 construction as test_core_eigenbasis.py: standardized covariance is
# (4/3) * [[1, 0.6], [0.6, 1]], eigenvalues 32/15 (along (1,1), high-variance)
# and 8/15 (along (1,-1), low-variance).
CORRELATED_CLEAN_X = np.array([[2.0, 2.0], [-2.0, -2.0], [1.0, -1.0], [-1.0, 1.0]])


def test_equal_magnitude_shift_scores_higher_along_low_variance_direction():
    """Core paper claim (Sec. 3): 'more effective attacks tend to cause larger
    shifts along directions where the variance is low.' A unit shift along the
    low-variance eigendirection must score higher than an equal-magnitude
    shift along the high-variance eigendirection.
    """
    model = fit_drs(CORRELATED_CLEAN_X, num_directions=2)

    high_variance_shift = np.array([1.0, 1.0]) / np.sqrt(2.0)
    low_variance_shift = np.array([1.0, -1.0]) / np.sqrt(2.0)

    high_score = drs_score(high_variance_shift, model)
    low_score = drs_score(low_variance_shift, model)

    assert low_score > high_score
    # Analytically: ratio = sqrt(eigenvalue_high / eigenvalue_low)
    #             = sqrt((32/15) / (8/15)) = 2
    assert low_score == pytest.approx(2.0 * high_score, rel=1e-6)


def test_perturbation_orthogonal_to_clean_manifold_is_detected():
    """Small-scale replication of Fig. 2(d): clean embeddings concentrate near
    a low-dimensional correlated manifold (as real neural embeddings do).
    Perturbations pushed off that manifold, into its low-variance orthogonal
    complement, are exactly what DRS is designed to flag.
    """
    rng = np.random.default_rng(7)
    d, k, n = 20, 3, 500  # d observed dims, k latent factors, n samples

    loading = rng.normal(size=(k, d))

    def sample_clean(n_samples, noise_std=0.05):
        latents = rng.normal(size=(n_samples, k))
        return latents @ loading + rng.normal(scale=noise_std, size=(n_samples, d))

    clean_fit = sample_clean(n)
    clean_holdout = sample_clean(n)

    model, _, threshold = fit_drs_with_threshold(clean_fit, num_directions=d - k, quantile=0.99)
    clean_flag_rate = is_flagged(drs_score(clean_holdout, model), threshold).mean()

    # Find a direction orthogonal to the manifold spanned by `loading` (its
    # right null space) and push clean points off the manifold along it.
    _, _, vt = np.linalg.svd(loading, full_matrices=True)
    off_manifold_direction = vt[k]  # first right-singular vector outside the row space of `loading`

    poisoned = sample_clean(n) + 3.0 * off_manifold_direction
    poison_flag_rate = is_flagged(drs_score(poisoned, model), threshold).mean()

    assert clean_flag_rate < 0.05
    assert poison_flag_rate > 0.9


def test_handles_zero_variance_dimension_without_nan_or_inf():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(30, 4))
    X[:, 2] = 5.0  # constant column: zero variance

    model = fit_drs(X, num_directions=4)
    scores = drs_score(X, model)

    assert np.isfinite(scores).all()
    assert model.num_directions == 4
```

- [ ] **Step 2: Run it**

```bash
pytest tests/test_paper_claims.py -v
```
Expected: PASS (3 tests) — no production code should need changes if Tasks 1-3 were implemented as specified. If `test_perturbation_orthogonal_to_clean_manifold_is_detected` is flaky, first check the constants (`noise_std=0.05` vs. shift magnitude `3.0`, `n=500`) before touching `core.py` — the separation margin is roughly 60x the clean noise scale, so failures indicate an implementation bug, not a flaky threshold.

- [ ] **Step 3: Run the full `drs_defense` suite together**

```bash
pytest -v
```
Expected: PASS (13 tests total across Tasks 1-4).

- [ ] **Step 4: Commit**

```bash
git add drs_defense/tests/test_paper_claims.py
git commit -m "test(drs_defense): verify paper's core qualitative DRS claims"
```

---

## Task 5: Package documentation

**Files:**
- Create: `drs_defense/README.md`

- [ ] **Step 1: Write the README**

`drs_defense/README.md`:
```markdown
# drs_defense

Reference implementation of the **DRS (Directional Relative Shifts)** poisoning
defense from *"Understanding Data Poisoning Attacks for RAG: Insights and
Algorithms"* (ICLR 2025 submission), https://openreview.net/pdf?id=2aL6gcFX7q.

This module exists because four independent reimplementations of DRS
(`Agent_Setting/ReAct/drs.py`, `RAG_Setting/src/medrag_repro/defense/drs.py`,
`RAG_Setting/src/medrag_repro/defense/drs_old.py`, and
`Retrieving_stage/poisonrag_experiment/drs.py`) had drifted from the paper's
formula and from each other. All DRS math in this repo now lives here; the
remaining per-project `drs.py` files are thin adapters over this package that
preserve each subproject's existing call signatures.

## The algorithm (paper Section 4, Algorithm 1 & Eq. 3)

Given a standardized clean embedding matrix `X` (zero mean, unit variance per
dimension):

1. Eigendecompose the covariance `S = V Λ V^T`.
2. Sort eigenvalues (and eigenvectors) **ascending**.
3. For any embedding `z`, over the `M` **smallest**-eigenvalue directions:

   `DRS(z; X) = Σ_{i=1}^{M} |z^T v_i| / sqrt(λ_i)`

Algorithm 2 (detection): fit on clean reference embeddings, set the decision
threshold `τ` to the `q`-th quantile of the clean DRS scores, and flag any
future embedding `z` with `DRS(z; X_clean) > τ`.

## Install

From the repo root, in whichever environment a subproject uses:

```bash
pip install -e ./drs_defense
```

or, from inside a subproject directory (matches this repo's existing
`-e .`-style requirements files):

```bash
pip install -e ../drs_defense
```

## API

```python
from drs_defense.core import fit_drs, drs_score, fit_drs_with_threshold, is_flagged

model = fit_drs(clean_embeddings, num_directions=100)          # Algorithm 1
scores = drs_score(embeddings, model)                          # Eq. 3

model, clean_scores, threshold = fit_drs_with_threshold(
    clean_embeddings, num_directions=100, quantile=0.99,
)                                                                # Algorithm 2
flagged = is_flagged(drs_score(candidates, model), threshold)
```

## Tests

```bash
pip install -e "./drs_defense[dev]"
pytest drs_defense/tests -q
```

The suite includes a hand-computed regression value tied directly to Eq. 3,
plus small synthetic experiments verifying the paper's qualitative claims:
shifts along low-variance directions score higher than equal-magnitude
shifts along high-variance directions, and perturbations pushed off a
correlated clean-data manifold are detected at the calibrated FPR.
```

- [ ] **Step 2: Commit**

```bash
git add drs_defense/README.md
git commit -m "docs(drs_defense): document Algorithm 1/2, install, and API"
```

---

## Task 6: Refactor `Agent_Setting/ReAct/drs.py` to delegate to `drs_defense`

**Files:**
- Modify: `Agent_Setting/ReAct/drs.py`
- Modify: `Agent_Setting/environment.yml`
- Test: `Agent_Setting/tests/test_drs_parity.py`

**Interfaces:**
- Consumes: `drs_defense.core.fit_drs_with_threshold`, `drs_defense.core.drs_score`, `drs_defense.core.DRSModel`.
- Produces: unchanged public surface — `fit_drs(clean_embeddings: torch.Tensor, num_directions=200, quantile=0.99) -> DRSStats`, `score_drs(embeddings: torch.Tensor, stats: DRSStats) -> torch.Tensor`, and `DRSStats` still exposes `.threshold`, `.false_positive_rate`, `.num_directions` (the only fields `local_wikienv.py` reads externally — confirmed by grep before writing this plan).

This delegation moves the eigendecomposition from GPU/PyTorch to CPU/NumPy (via `drs_defense.core`). This is a one-time fit over a modest number of reference embeddings (not a hot loop), so the performance impact is negligible; it removes the previous implicit reliance on `eigh`'s ascending-order guarantee (now explicit via `argsort` in `low_variance_eigenbasis`).

- [ ] **Step 1: Write the failing parity test**

Create `Agent_Setting/tests/test_drs_parity.py`:
```python
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd Agent_Setting
pip install -e ../drs_defense
pip install pytest  # if not already present in this environment
pytest tests/test_drs_parity.py -v
```
Expected: FAIL — current `ReAct/drs.py` has no bug given the same inputs (formula already matches the paper), but the eigenvector-sign/ordering comparison and `DRSStats` internals will not yet be backed by `drs_defense`, so this test is really checking today's implementation stays correct through the refactor. It should already pass against the *old* file's math; if it does, that's fine — proceed to Step 3 to make the delegation real, then re-run to confirm it still passes with the new implementation wired in.

- [ ] **Step 3: Replace `Agent_Setting/ReAct/drs.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from drs_defense.core import DRSModel
from drs_defense.core import drs_score as _drs_score_np
from drs_defense.core import fit_drs_with_threshold


@dataclass
class DRSStats:
    mean: torch.Tensor
    std: torch.Tensor
    eigenvectors: torch.Tensor
    eigenvalues: torch.Tensor
    threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float
    num_directions: int
    _np_model: DRSModel = field(repr=False)


def fit_drs(clean_embeddings: torch.Tensor, num_directions: int = 200, quantile: float = 0.99) -> DRSStats:
    if clean_embeddings.dim() != 2:
        raise ValueError("clean_embeddings must be a 2D tensor")

    device = clean_embeddings.device
    clean_np = clean_embeddings.detach().cpu().double().numpy()

    model, clean_scores_np, threshold = fit_drs_with_threshold(
        clean_np, num_directions=num_directions, quantile=quantile,
    )

    return DRSStats(
        mean=torch.from_numpy(model.mean).float().to(device),
        std=torch.from_numpy(model.std).float().to(device),
        eigenvectors=torch.from_numpy(model.eigenvectors).float().to(device),
        eigenvalues=torch.from_numpy(model.eigenvalues).float().to(device),
        threshold=threshold,
        clean_scores=torch.from_numpy(clean_scores_np).float().to(device),
        false_positive_rate=float((clean_scores_np > threshold).mean()),
        num_directions=model.num_directions,
        _np_model=model,
    )


def score_drs(embeddings: torch.Tensor, stats: DRSStats) -> torch.Tensor:
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)

    device = embeddings.device
    scores_np = _drs_score_np(embeddings.detach().cpu().double().numpy(), stats._np_model)
    return torch.from_numpy(scores_np).float().to(device)
```

- [ ] **Step 4: Wire the dependency into the conda environment**

In `Agent_Setting/environment.yml`, add a line to the `pip:` list (after `- bs4`):
```yaml
    - bs4
    - -e ../drs_defense
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_drs_parity.py -v
```
Expected: PASS (2 tests).

- [ ] **Step 6: Commit**

```bash
git add Agent_Setting/ReAct/drs.py Agent_Setting/environment.yml Agent_Setting/tests/test_drs_parity.py
git commit -m "refactor(Agent_Setting): delegate DRS math to drs_defense"
```

---

## Task 7: Refactor `RAG_Setting/src/medrag_repro/defense/drs.py` (`DRSDetector`)

**Files:**
- Modify: `RAG_Setting/src/medrag_repro/defense/drs.py`
- Modify: `RAG_Setting/requirements.txt`
- Test: `RAG_Setting/tests/test_drs_detector_parity.py`

**Interfaces:**
- Consumes: `drs_defense.core.fit_drs`, `drs_defense.core.drs_score`, `drs_defense.core.DRSModel`.
- Produces: unchanged public surface — `DRSDetector(encoder, M=100, clean_quantile=0.99, eps=1e-8)` with `.fit(clean_texts)`, `.score_texts(texts)`, and (inherited from `BaseDetector`) `.detect(texts)`, `.upper_threshold`. New: `.clean_scores` and `.model` attributes (used by Task 8).

- [ ] **Step 1: Write the failing parity test**

Create `RAG_Setting/tests/test_drs_detector_parity.py`:
```python
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd RAG_Setting
pip install -e ../drs_defense
pip install pytest  # if not already present in this environment
pytest tests/test_drs_detector_parity.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'drs_defense'` (until Step 4 installs it) or an assertion mismatch against the pre-refactor implementation's internals.

- [ ] **Step 3: Replace `RAG_Setting/src/medrag_repro/defense/drs.py`**

```python
from __future__ import annotations

from typing import Sequence

import numpy as np
from drs_defense.core import DRSModel, drs_score, fit_drs

from medrag_repro.defense.common import BaseDetector
from medrag_repro.retriever.contriever import ContrieverEncoder


class DRSDetector(BaseDetector):
    """DRS defense (paper Algorithm 1 & 2); math delegated to drs_defense.core."""

    def __init__(self, encoder: ContrieverEncoder, M: int = 100, clean_quantile: float = 0.99, eps: float = 1e-8):
        super().__init__(two_sided=False, upper_quantile=clean_quantile)
        self.encoder = encoder
        self.M = M
        self.eps = eps
        self.model: DRSModel | None = None
        self.clean_scores: np.ndarray | None = None

    def fit(self, clean_texts: Sequence[str]) -> None:
        X = self.encoder.encode(list(clean_texts), normalize=False).astype(np.float64)
        self.model = fit_drs(X, num_directions=self.M, eps=self.eps)
        self.clean_scores = drs_score(X, self.model)
        self.fit_thresholds_from_scores(self.clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        assert self.model is not None
        Z = self.encoder.encode(list(texts), normalize=False).astype(np.float64)
        return drs_score(Z, self.model)
```

- [ ] **Step 4: Wire the dependency into requirements**

In `RAG_Setting/requirements.txt`, add a line:
```
-e .
-e ../drs_defense
pytest
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_drs_detector_parity.py -v
```
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add RAG_Setting/src/medrag_repro/defense/drs.py RAG_Setting/requirements.txt \
        RAG_Setting/tests/test_drs_detector_parity.py
git commit -m "refactor(RAG_Setting): delegate DRSDetector math to drs_defense"
```

---

## Task 8: Retire `drs_old.py`; migrate `run_drs.py` to `DRSDetector`

**Files:**
- Modify: `RAG_Setting/scripts/run_drs.py`
- Delete: `RAG_Setting/src/medrag_repro/defense/drs_old.py`

**Interfaces:**
- Consumes: `medrag_repro.defense.drs.DRSDetector` (Task 7) — specifically `.fit()`, `.score_texts()`, `.detect()`, `.clean_scores`, `.upper_threshold`.

`drs_old.py`'s `compute_drs_reference` is a second, independently-diverged copy of the exact same math `DRSDetector` already implements (same encoder-fit-score pattern), still wired into `run_drs.py` as a separate pipeline step from `run_defense.py --method drs`. Since `DRSDetector` now covers this fully, retire the duplicate rather than adapting it a fourth time.

- [ ] **Step 1: Rewrite `RAG_Setting/scripts/run_drs.py` to use `DRSDetector`**

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from medrag_repro.config import load_config
from medrag_repro.datamodels import CorpusDoc, PoisonDoc, QAItem
from medrag_repro.defense.drs import DRSDetector
from medrag_repro.evaluation.rag_eval import evaluate_attack
from medrag_repro.llm.client import load_openai_client
from medrag_repro.retriever.contriever import ContrieverEncoder
from medrag_repro.retriever.index import load_index, retrieve_topk
from medrag_repro.utils.io import read_jsonl, write_json, write_jsonl
from medrag_repro.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    client = load_openai_client()
    clean_queries = [QAItem(**row) for row in read_jsonl(cfg["paths"]["clean_queries"])]
    targets = [QAItem(**row) for row in read_jsonl(cfg["paths"]["targets"])]
    corpus = [CorpusDoc(**row) for row in read_jsonl(cfg["paths"]["pubmed_corpus"])]
    poison = [PoisonDoc(**row) for row in read_jsonl(cfg["paths"]["poison_docs"])]
    doc_lookup = {d.doc_id: d.text for d in corpus}
    index, doc_ids, vectors = load_index(cfg["paths"]["index_dir"])
    encoder = ContrieverEncoder(
        model_name=cfg["retriever"]["model_name"],
        batch_size=cfg["retriever"]["batch_size"],
        device=cfg["retriever"]["device"],
    )

    clean_ref_map = {}
    for qa in clean_queries:
        retrieved = retrieve_topk(qa.question, encoder, index, doc_ids, doc_lookup, cfg["retriever"]["top_k"])
        for doc_id, _, text in retrieved:
            clean_ref_map.setdefault(doc_id, text)
    clean_ref_texts = list(clean_ref_map.values())

    detector = DRSDetector(
        encoder=encoder,
        M=cfg["drs"]["M"],
        clean_quantile=cfg["drs"]["clean_threshold_quantile"],
    )
    detector.fit(clean_ref_texts)

    poison_texts = [p.full_text for p in poison]
    poison_scores = detector.score_texts(poison_texts)
    detected = detector.detect(poison_texts)
    kept_poison = [p for p, d in zip(poison, detected) if not d]

    clean_scores = detector.clean_scores
    threshold = detector.upper_threshold

    post = evaluate_attack(
        client=client,
        model=cfg["llm_eval"]["answer_model"],
        encoder=encoder,
        index=index,
        doc_ids=doc_ids,
        vectors=vectors,
        doc_lookup=doc_lookup,
        targets=targets,
        poison_docs=kept_poison,
        k=cfg["retriever"]["top_k"],
        answer_temperature=cfg["llm_eval"]["answer_temperature"],
    )

    details = [
        {"poison_id": p.poison_id, "target_qid": p.target_qid, "score": float(s), "detected": bool(d)}
        for p, s, d in zip(poison, poison_scores.tolist(), detected)
    ]
    out = {
        "n_clean_reference_docs": len(clean_ref_texts),
        "threshold": threshold,
        "empirical_clean_fpr": float(np.mean(clean_scores > threshold)),
        "poison_detection_rate": float(np.mean(detected)) if detected else 0.0,
        "n_poison_before": len(poison),
        "n_poison_after": len(kept_poison),
        "post_defense_attack_success_rate": post["attack_success_rate"],
        "post_defense_retrieval_precision": post["retrieval_precision"],
        "post_defense_retrieval_recall": post["retrieval_recall"],
        "post_defense_retrieval_f1": post["retrieval_f1"],
        "details": details,
    }
    write_json(cfg["paths"]["drs_metrics"], out)
    write_jsonl(str(cfg["paths"]["artifact_dir"]) + "/kept_poison_after_drs.jsonl", kept_poison)
    print({k: v for k, v in out.items() if k != "details"})


if __name__ == "__main__":
    main()
```

Note this preserves the exact original output JSON schema of `run_drs.py` (`n_clean_reference_docs`, `threshold`, `empirical_clean_fpr`, `poison_detection_rate`, `n_poison_before`, `n_poison_after`, `post_defense_*`, `details`), so anything consuming `cfg["paths"]["drs_metrics"]` downstream is unaffected.

- [ ] **Step 2: Delete the retired file**

```bash
git rm RAG_Setting/src/medrag_repro/defense/drs_old.py
```

- [ ] **Step 3: Confirm nothing else imports it**

```bash
grep -rn "drs_old\|compute_drs_reference" RAG_Setting/ --include='*.py'
```
Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add RAG_Setting/scripts/run_drs.py
git commit -m "refactor(RAG_Setting): retire drs_old.py, migrate run_drs.py to DRSDetector"
```

---

## Task 9: Refactor `Retrieving_stage/poisonrag_experiment/drs.py`

**Files:**
- Modify: `Retrieving_stage/poisonrag_experiment/drs.py`
- Modify: `Retrieving_stage/requirements.txt`
- Test: `Retrieving_stage/tests/test_drs_parity.py`

**Interfaces:**
- Consumes: `drs_defense.core.DRSModel`, `drs_defense.core.standardize`, `drs_defense.core.low_variance_eigenbasis`, `drs_defense.core.drs_score`.
- Produces: unchanged public surface — `fit_drs(clean_embeddings, num_directions=32, power=1.0, eps=1e-8) -> dict`, `drs_score(embedding, drs_model) -> float`, `drs_threshold(clean_embeddings, quantile=0.99, num_directions=32, power=1.0) -> (model, clean_scores, threshold)`. The `power` ablation knob (only used with non-default values via `run_poisonrag_experiment.py --drs_power`) is preserved locally since it is not part of the paper's formula and doesn't belong in the shared paper-faithful `drs_defense.core` module; at its default `power=1.0` the function fully delegates to `drs_defense.core`.

- [ ] **Step 1: Write the failing parity test**

Create `Retrieving_stage/tests/test_drs_parity.py`:
```python
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
```

- [ ] **Step 2: Run it to verify it fails**

```bash
cd Retrieving_stage
pip install -e ../drs_defense
pip install pytest  # if not already present in this environment
pytest tests/test_drs_parity.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'drs_defense'` until installed, then an assertion gap versus the pre-refactor float32-only implementation.

- [ ] **Step 3: Replace `Retrieving_stage/poisonrag_experiment/drs.py`**

```python
from __future__ import annotations

import numpy as np
from drs_defense.core import DRSModel
from drs_defense.core import drs_score as _core_drs_score
from drs_defense.core import low_variance_eigenbasis, standardize


def fit_drs(clean_embeddings, num_directions=32, power=1.0, eps=1e-8):
    """
    Fit a DRS model on clean embeddings.

    DRS emphasizes shifts along low-variance directions in the clean data
    (paper Eq. 3: https://openreview.net/pdf?id=2aL6gcFX7q). `power` is a
    local ablation knob (not part of the paper's formula); leave it at the
    default 1.0 to reproduce Eq. 3 exactly.
    """
    clean_embeddings = np.asarray(clean_embeddings, dtype=np.float32)
    if clean_embeddings.ndim != 2:
        raise ValueError("clean_embeddings must be a 2D array.")
    if len(clean_embeddings) < 2:
        raise ValueError("At least two clean embeddings are required for DRS.")

    standardized, mean, std = standardize(clean_embeddings, eps=eps)
    eigenvalues, eigenvectors = low_variance_eigenbasis(standardized, num_directions)

    return {
        "mean": mean,
        "std": std,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "power": power,
        "eps": eps,
    }


def drs_score(embedding, drs_model):
    """Compute DRS for a single embedding."""
    if drs_model["power"] == 1.0:
        model = DRSModel(
            mean=drs_model["mean"],
            std=drs_model["std"],
            eigenvalues=drs_model["eigenvalues"],
            eigenvectors=drs_model["eigenvectors"],
            num_directions=drs_model["eigenvectors"].shape[1],
            eps=drs_model["eps"],
        )
        return float(_core_drs_score(embedding, model))

    z = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
    z_std = (z - drs_model["mean"]) / drs_model["std"]
    z_std = z_std[0]
    projections = np.abs(z_std @ drs_model["eigenvectors"]) ** drs_model["power"]
    denom = np.sqrt(np.maximum(drs_model["eigenvalues"], drs_model["eps"]))
    return float(np.sum(projections / denom))


def drs_threshold(clean_embeddings, quantile=0.99, num_directions=32, power=1.0):
    """Fit DRS and derive a threshold from clean samples."""
    model = fit_drs(
        clean_embeddings=clean_embeddings,
        num_directions=num_directions,
        power=power,
    )
    clean_scores = [drs_score(embedding, model) for embedding in clean_embeddings]
    threshold = float(np.quantile(clean_scores, quantile))
    return model, clean_scores, threshold
```

Note: `mean`/`std`/`eigenvalues`/`eigenvectors` in the returned dict are now `float64` (via `drs_defense.core.standardize`/`low_variance_eigenbasis`) instead of the previous `float32`. This is a precision *increase*, not a correctness regression, and brings this file in line with the other two subprojects, which already computed in `float64`.

- [ ] **Step 4: Wire the dependency into requirements**

In `Retrieving_stage/requirements.txt`, add two lines at the end:
```
-e ../drs_defense
pytest
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
pytest tests/test_drs_parity.py -v
```
Expected: PASS (3 tests).

- [ ] **Step 6: Commit**

```bash
git add Retrieving_stage/poisonrag_experiment/drs.py Retrieving_stage/requirements.txt \
        Retrieving_stage/tests/test_drs_parity.py
git commit -m "refactor(Retrieving_stage): delegate DRS math to drs_defense, keep power ablation"
```

---

## Task 10: Document the shared module in repo-level docs

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:** None (docs only).

- [ ] **Step 1: Add `drs_defense` to the root `README.md` subproject list**

After the `Agent_Setting/` bullet in the "Subprojects" list, add:
```markdown
- [`drs_defense/`](drs_defense/README.md) — shared reference implementation of the DRS (Directional Relative Shifts) poisoning defense (Algorithm 1 & Eq. 3 of the DRS paper), used by all three subprojects above instead of each maintaining its own copy.
```

- [ ] **Step 2: Update `CLAUDE.md`**

In the "Repository overview" bullet list, add a `drs_defense/` entry alongside the three subprojects:
```markdown
- `drs_defense/` — shared, pip-installable reference implementation of the DRS (Directional Relative Shifts) poisoning defense (paper Algorithm 1 & Eq. 3, https://openreview.net/pdf?id=2aL6gcFX7q). `Agent_Setting/ReAct/drs.py`, `RAG_Setting/src/medrag_repro/defense/drs.py`, and `Retrieving_stage/poisonrag_experiment/drs.py` are thin adapters over this module — do not reimplement DRS math locally; add it here and delegate.
```

Replace the existing "No tests or linters" section (since this is no longer accurate) with:
```markdown
## Tests

`drs_defense/` has a pytest suite (`drs_defense/tests/`) verifying the DRS implementation against the paper's Algorithm 1/2 and Eq. 3, plus small parity test suites in each subproject (`Agent_Setting/tests/`, `RAG_Setting/tests/`, `Retrieving_stage/tests/`) that check their DRS adapters match `drs_defense.core` exactly. Everything else in the repo still has no test suite, CI config, or linter/formatter — verify other changes by running the relevant script(s) end-to-end against small/sample data.
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: document the shared drs_defense module"
```

---

## Self-review

**Spec coverage:**
- "Move DRS functions into a separate subfolder, usable as a module" → Tasks 1-5 (new `drs_defense/` package, installable via `pip install -e`).
- "Extra supporting functions" → `fit_drs_with_threshold`, `is_flagged`, `quantile_threshold` (Task 3) on top of the core `fit_drs`/`drs_score` (Task 2).
- "Testing functions to verify correctness based on its paper" → Task 1 (ascending-order/eigenbasis-selection tests), Task 2 (hand-computed Eq. 3 regression test), Task 3 (Algorithm 2 threshold-calibration test), Task 4 (the paper's qualitative claims: low-variance-direction sensitivity, off-manifold detection, degenerate-input robustness).
- All four existing DRS call sites migrated and parity-tested: Task 6 (Agent_Setting), Task 7 (RAG_Setting `DRSDetector`), Task 8 (RAG_Setting `drs_old.py`/`run_drs.py`), Task 9 (Retrieving_stage).
- Documentation of the new module and its role: Task 5 (package README) and Task 10 (repo-level docs).

**Placeholder scan:** none — every step has complete file contents or exact diffs; no "TODO"/"similar to Task N"/"add tests for the above" left unresolved.

**Type consistency:** `DRSModel(mean, std, eigenvalues, eigenvectors, num_directions, eps)` is defined once in Task 2 and used with identical field names in Tasks 6, 7, and 9. `fit_drs`/`drs_score`/`fit_drs_with_threshold`/`is_flagged`/`quantile_threshold` signatures introduced in Tasks 2-3 are consumed unchanged by every later task.

---

**Plan complete and saved to `docs/superpowers/plans/2026-08-07-drs-shared-module.md`.**
