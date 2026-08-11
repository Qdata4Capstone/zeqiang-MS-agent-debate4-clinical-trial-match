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

### Few reference samples relative to embedding dimensionality (n ≤ d)

Step 1's covariance `S` is `d × d`. If the clean reference set has `n ≤ d`
samples (e.g. 20-200 reference documents against a 768-dim MedCPT/Contriever
embedding — a real, not hypothetical, case: confirmed against a live
`trial_retrieval` run), `S` is rank-deficient: `d - rank(X)` of its
eigenvalues are *exact* numerical zeros, since mean-centered `n × d` data has
rank at most `n - 1`. Naively eigendecomposing the full `d × d` matrix and
picking the smallest `M` eigenvalues picks these up first — they aren't
genuinely low-variance directions, they're directions the reference set has
*no data in at all*. `DRS(z; X)`'s `1/√λ_i` term then explodes for almost any
out-of-sample `z` along them, while `X`'s own reference points score ~0 there
by construction (their own covariance was fit to make that exactly true).
Confirmed at real scale: with `n=20`, `d=768`, clean reference scores landed
around `1e-10` and unseen-candidate scores around `1e5` — DRS flagged
55-72% of an entire corpus as poisoned, worse than no defense at all.

`low_variance_eigenbasis` routes to **dual (Gram-matrix) PCA** whenever
`n <= d`: eigendecompose the `n × n` matrix `X Xᵀ` instead of the `d × d`
covariance `Xᵀ X`. They share the exact same nonzero eigenvalues (if
`X = U S Vᵀ` is the SVD, `X Xᵀ = U S² Uᵀ` and `Xᵀ X = V S² Vᵀ`), so this
recovers the identical low-variance directions the covariance would have
given — but caps the count at the data's true rank (`≤ n - 1`) with no
spurious zeros mixed in, and recovers the `d`-dimensional eigenvectors via
`v_i = Xᵀ u_i / √(λ_i (n - 1))`. See
[stats.stackexchange.com/questions/7111](https://stats.stackexchange.com/questions/7111/how-to-perform-pca-for-data-of-very-high-dimensionality)
for the general technique.

This eliminates the catastrophic false-positive blowup, but not DRS's
underlying, expected statistical-power limitation with a small reference
set — a `768`-dim embedding space still benefits from `n` closer to or above
`768` for *detecting* subtle poisoning (fewer reference samples means a
noisier, less powerful eigenbasis, not a broken one). Use as large a clean
reference set as your data reasonably supports; see
`use-cases/trial_retrieval/poisonrag_experiment/README.md`'s `--drs_ref_k`
note for real before/after numbers at two reference-set sizes, and
[`docs/drs-dual-pca-analysis.md`](../docs/drs-dual-pca-analysis.md) for the
full writeup — including real numbers on how this changes DRS's standing
against the L2-norm/L2-distance/perplexity baselines (it now wins in one
use case and loses in another, depending on reference-set size).

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
small synthetic experiments verifying the paper's qualitative claims (shifts
along low-variance directions score higher than equal-magnitude shifts along
high-variance directions, and perturbations pushed off a correlated
clean-data manifold are detected at the calibrated FPR), and
`test_core_dual_pca.py`, covering the `n <= d` dual-PCA path: exact
equivalence with the primal covariance's real (non-spurious) eigenvalues,
correct rank-capping when more directions are requested than the data
supports, and a regression check against the real catastrophic-blowup shape
(`n=20`, `d=200`) confirming an unseen point no longer scores orders of
magnitude above the clean reference spread.
