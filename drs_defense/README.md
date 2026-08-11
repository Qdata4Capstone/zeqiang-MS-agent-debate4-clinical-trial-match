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

### Choosing `M` (`num_directions`) and reference-set size `n`

`n` (how many clean reference documents you fit on) and `M` (how many
low-variance directions `drs_score` sums over, Eq. 3) both have to scale up
*together* — pushing only one of them does little on its own. This is
directly testable with `use-cases/medqa_rag/scripts/sweep_reference_size.py`,
which sweeps both against poison-detection rate and clean FPR without
needing any LLM calls. Real results from that sweep (`medqa_rag`, 3 poison
docs, `configs/sweep.yaml`'s small local corpus — see
[`docs/drs-dual-pca-analysis.md`](../docs/drs-dual-pca-analysis.md)'s
"Crossover confirmed" section for the full table):

| Pooled reference docs (`n`) | `M=100` | `M=200` | `M=300` |
|---|---|---|---|
| 241 | 0.67 | 0.33 | 0.33 (clipped to `n-1`) |
| 326 | **1.00** | **1.00** | **1.00** |

At `n=241`, `M=100` (the paper's own value) beats larger `M` — with a
reference set this small, `M=200`/`300` are asking for more directions than
the data can estimate reliably. At `n=326`, every `M` from 100-300 reaches
perfect detection: once `n` is large enough, DRS stops being sensitive to
the exact value of `M`. The baselines (L2-norm, L2-distance, perplexity)
stayed at 0.00-0.33 across every `(n, M)` cell in that sweep — this crossover
is what makes DRS worth using at all, and it doesn't show up if `n` is left
at demo scale (`use-cases/medqa_rag/configs/demo.yaml`'s `n=29` never
detects anything, at any `M`).

Rules of thumb, in order of impact:

1. **Start with `M=100`** — every table in the paper's main text (Section
   5.1.1, Tables 2-5) uses this value; it's also the tested value in the
   sweep above. Don't reach for a larger `M` before `n` is large — the
   table above shows that backfires (`n=241, M=200` is *worse* than
   `n=241, M=100`).
2. **Pool the reference set across every protected query** (the paper's
   actual Algorithm 2) rather than fitting one DRS model per query. This
   alone was the difference between catching 1/3 and 3/3 poison docs in
   `trial_retrieval` at the same `--drs_ref_k` — see that README's
   "Per-query vs. pooled reference sets" table. `medqa_rag`'s
   `run_defense.py` and `strategyqa_agent`'s `_fit_drs` already pool by
   construction; `trial_retrieval` defaults to pooling but can opt out with
   `--no-drs_pool_reference`.
3. **Grow `n` before growing `M`.** `M` is capped by both `M <= d`
   (embedding dimensionality) and, via the dual-PCA path above, `M <= n-1`
   — a small reference set structurally limits how large `M` can even be,
   independent of whether a larger `M` would help. The sweep above found
   `n` needs to be at least a few hundred documents (not embedding-space
   dimensionality `d` itself — `n=326 << d=768` already reached perfect
   detection here) before `M=100` has enough to work with; below that, more
   `M` doesn't compensate.
4. **When in doubt, measure it.** `n` needed for good detection depends on
   the embedding model, corpus, and attack — the numbers above are one data
   point, not a universal threshold. Run
   `use-cases/medqa_rag/scripts/sweep_reference_size.py --config <your
   config> --ref_sizes <comma list> --m_values <comma list>` against your
   own setup to find where detection actually crosses over the baselines,
   rather than assuming these numbers transfer.

### Caveats on n and M: what these numbers don't tell you

The guidance above shows *that* `n` and `M` matter and roughly how, but the
specific numbers come with real limitations. Don't take them further than
the evidence supports:

- **The detection-rate numbers above have very low resolution.** Every
  sweep in this repo used 3 poison documents, so "detection rate" only
  takes values `{0.00, 0.33, 0.67, 1.00}` — one document flipping detected/
  not-detected moves the whole number by 33 points. The non-monotonic dip
  at `ref_size=500, M=300` (1.00 at `M=200`/`250`, back to 0.67 at `M=300`
  — see `docs/drs-dual-pca-analysis.md`) is most likely exactly this kind
  of single-document noise, not a real "`M=300` is worse than `M=250`"
  effect — but with only 3 trials there's no way to tell noise from signal
  apart. Don't tune `M` to the last few percentage points against a
  poison set this small; re-run with more poison documents (or repeat
  with different random seeds) before trusting a difference of one
  detection.
- **`M` and `n` thresholds are not portable constants.** `n=326`/`M=100`
  reaching 1.00 here is specific to `medqa_rag`'s Contriever embeddings,
  its 1,500-doc PubMed corpus, and its particular PoisonedRAG-generated
  poison docs. A different embedding model, corpus, or attack could need a
  larger or smaller `n` for the same `M` — rule 4 above (measure it on your
  own setup) isn't a formality, it's load-bearing.
- **Pooling more clean queries has a corpus-size ceiling, not a query-count
  ceiling.** Growing `medqa.n_clean_queries` from 300 to 600 only grew the
  pooled reference set from 241 to 326 unique docs (both draw top-`k=5`
  from the same 1,500-doc corpus, so most retrieved docs were already
  seen). If your corpus is small relative to how many clean queries you
  can generate, adding more queries stops helping well before `n` reaches
  the target you actually need — grow the corpus, not just the query
  count, if you hit this ceiling.
- **DRS assumes the reference set is actually clean.** Every number above
  fits DRS on a reference set known by construction to contain no poison.
  In a real deployment, the "clean" query set is usually just *assumed*
  clean (e.g., historical queries that weren't flagged), not verified — if
  poisoned documents are already in it, they corrupt the eigenbasis DRS
  fits on and the defense silently degrades with no signal that anything
  is wrong. This isn't unique to DRS (every reference-based baseline here
  shares the assumption), but DRS's whole mechanism is a function of the
  reference set's estimated covariance structure, so it has more surface
  area for a contaminated reference set to distort than a single global
  statistic like L2-norm does.
- **A larger reference set costs more, not just detects more.** Fitting DRS
  is an eigendecomposition of an `n x n` (dual-PCA, `n <= d`) or `d x d`
  (primal, `n > d`) matrix, plus encoding every reference document with the
  retrieval embedder. Both scale up with `n` — cheap at the `n` in the
  hundreds used here, but worth budgeting for explicitly if a deployment's
  "as large a reference set as you can support" (per the guidance above)
  means tens of thousands of documents, not hundreds.
- **The quantile threshold gets noisier at small `n`.** `clean_threshold_quantile: 0.99`
  sets the flagging threshold to the 99th percentile of clean DRS scores —
  with `n` in the low hundreds, that's an estimate from only a handful of
  the most extreme clean samples (e.g. ~3 samples above the 99th
  percentile at `n=300`). The threshold itself, not just the eigenbasis,
  is less reliable at small `n`; the clean-FPR numbers reported throughout
  this repo's sweeps (consistently <=0.04) are real but come with the same
  small-sample caveat as the detection-rate numbers above.

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
