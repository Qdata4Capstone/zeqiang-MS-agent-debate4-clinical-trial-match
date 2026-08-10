# drs_defense

Reference implementation of the **DRS (Directional Relative Shifts)** poisoning
defense from *"Understanding Data Poisoning Attacks for RAG: Insights and
Algorithms"* (ICLR 2025 submission), https://openreview.net/pdf?id=2aL6gcFX7q.

This module exists because four independent reimplementations of DRS
(`use-cases/strategyqa_agent/ReAct/drs.py`, `use-cases/medqa_rag/src/medrag_repro/defense/drs.py`,
`use-cases/medqa_rag/src/medrag_repro/defense/drs_old.py` (since retired), and
`use-cases/trial_retrieval/poisonrag_experiment/drs.py`) had drifted from the paper's
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
pip install -e ../../drs_defense
```

## Code structure

```
drs_defense/
  src/drs_defense/
    __init__.py
    core.py     # standardize, low_variance_eigenbasis, DRSModel, fit_drs, drs_score,
                 # quantile_threshold, fit_drs_with_threshold, is_flagged
  tests/        # pytest suite (Algorithm 1/2, Eq. 3 regression + qualitative checks)
```

Single-module package — all DRS math lives in `core.py`.

## Quick start

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
