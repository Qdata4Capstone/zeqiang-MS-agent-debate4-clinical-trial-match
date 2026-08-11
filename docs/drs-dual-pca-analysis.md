# DRS: Dual-PCA Fix and Defense Comparison Analysis

**Date:** 2026-08-10
**Code:** `drs_defense/src/drs_defense/core.py` (`low_variance_eigenbasis`), tests in `drs_defense/tests/test_core_dual_pca.py`
**Related docs:** `drs_defense/README.md` ("Few reference samples relative to embedding dimensionality" section), `use-cases/trial_retrieval/poisonrag_experiment/README.md` ("DRS use here" section)

## Summary

DRS (Directional Relative Shifts) catastrophically over-flagged legitimate
documents — recall *worse than no defense at all* — whenever its clean
reference set had fewer samples (`n`) than the embedding dimensionality
(`d`), which is the common case for real embedding models (MedCPT/Contriever
are both 768-dim) unless the reference set is deliberately large. Root
cause: the reference covariance matrix is rank-deficient in that regime, and
DRS's own algorithm (pick the smallest-eigenvalue directions) preferentially
selects the spurious, purely-numerical zero eigenvalues that produces. Fixed
by switching to dual (Gram-matrix) PCA when `n <= d` — a standard technique
(see [stats.stackexchange.com/questions/7111](https://stats.stackexchange.com/questions/7111/how-to-perform-pca-for-data-of-very-high-dimensionality))
that recovers the exact same real eigenvalues without the spurious ones.

The fix eliminates the false-positive catastrophe. It does **not** eliminate
DRS's real, expected loss of detection power when the reference set is
small — that's inherent to fitting any covariance-based estimator from few
samples, not a bug. Whether DRS outperforms the baseline defenses (L2-norm,
L2-distance, perplexity) after the fix now genuinely depends on reference-set
size relative to embedding dimensionality, confirmed with opposite outcomes
in this repo's two retrieval-based use cases (details below).

## Root cause

`low_variance_eigenbasis` (Algorithm 1, paper Section 4) eigendecomposes the
`d x d` covariance of the standardized clean reference embeddings and keeps
the `M` smallest-eigenvalue directions. If the reference set has `n`
samples, mean-centered `n x d` data has rank at most `n - 1`. When `n <= d`
(e.g. `--drs_ref_k 20` against 768-dim MedCPT embeddings), `d - (n - 1)` of
the covariance's eigenvalues are *exact* numerical zeros — directions the
reference set has literally no data in, not genuinely low-variance ones.

`drs_score`'s `1/sqrt(eigenvalue)` term (Eq. 3) then explodes for almost any
point along those directions, *except* the reference points themselves,
which score ~0 there by construction (a zero-eigenvalue direction of a
sample covariance has zero variance across exactly those samples). Net
effect: the clean reference set fits itself perfectly and flags nearly
everything else, including other legitimately clean documents.

Confirmed against a real run — `trial_retrieval`'s `poisonrag_experiment`,
`--drs_ref_k 20` (the script's own default), `n=20` clean reference
documents against MedCPT's `d=768`:

| | before fix |
|---|---|
| clean reference scores | ~1e-10 |
| unseen-candidate scores | ~1e5 |
| corpus flagged as poisoned | 55-72% (7,239 / ~10,872 candidates across 3 target patients) |
| recall@50/100/200 under DRS | 0.4013 / 0.4013 / 0.4013 (undefended baseline: 0.7052 / 0.8941 / 0.9137 — DRS made retrieval *worse*) |

## The fix

`low_variance_eigenbasis` now branches on `n <= d`. In that regime it
eigendecomposes the `n x n` Gram matrix `X @ X.T` instead of the `d x d`
covariance `X.T @ X`. If `X = U S V.T` (SVD), then `X @ X.T = U S^2 U.T` and
`X.T @ X = V S^2 V.T` — the two share the exact same nonzero eigenvalues,
and the `d`-dimensional eigenvectors are recovered via
`v_i = X.T @ u_i / sqrt(lambda_i * (n - 1))`. This caps the number of
directions at the data's true rank (`<= n - 1`) with no spurious zeros mixed
in — a numerical-linear-algebra identity, not an approximation.

When `n > d` (the regime every pre-existing test exercised), behavior is
byte-for-byte unchanged — the original primal covariance path still runs.

## Verification

**Synthetic (`drs_defense/tests/test_core_dual_pca.py`, 4 new tests, all
passing alongside the 14 pre-existing ones):**
- Dual-path eigenvalues/eigenvectors match the primal covariance's real
  (non-spurious) spectrum to float precision (max diff `8.9e-15` in one
  concrete check).
- Requesting more directions than the data supports correctly clips to
  `n - 1`, not to `d`.
- A direct repro of the bug's shape (`n=20`, `d=200`): before the fix this
  produced a ~1e14 score ratio between clean and unseen points; after the
  fix, unseen points (drawn from the same distribution as "clean" — no real
  poisoning in this synthetic case) score *lower* on average.

**Real-world, `trial_retrieval` (reusing cached MedCPT embeddings from the
original bug-finding run, zero re-encoding):**

| `--drs_ref_k` | metric | before fix | after fix |
|---|---|---|---|
| 20 | candidates flagged (3 queries) | 7,239 | 5 |
| 20 | recall@50/100/200 | 0.4013 / 0.4013 / 0.4013 | 0.7052 / 0.8941 / 0.9137 (= undefended baseline) |
| 20 | poison docs caught (of 3) | n/a (not checked pre-fix; over-flagging made it moot) | 0/3 |
| 200 | recall@50/100/200 | 0.6301 / 0.8190 / 0.8386 | 0.7052 / 0.8941 / 0.9137 (= undefended baseline) |
| 200 | poison docs caught (of 3) | not checked | 1/3 |

**Real-world, `medqa_rag` (`configs/demo.yaml`, `n=29` clean reference
docs against Contriever's `d=768`):**

| metric | before fix | after fix |
|---|---|---|
| poison detection rate | 1.0000 (3/3) | 0.0000 (0/3) |
| post-defense retrieval F1 | 0.0000 | 0.3333 (= undefended baseline) |

The pre-fix "1.0000 detection rate" was the same bug in different clothes:
at `n=29 << d=768`, the degenerate eigenbasis flagged essentially everything
handed to it, poison included — not genuine detection.

## DRS vs. baseline defenses: which wins now?

**It depends on reference-set size relative to embedding dimensionality —
confirmed with opposite outcomes in this repo's two retrieval use cases.**

**`trial_retrieval` at `--drs_ref_k 200`** (out of a 3,624-doc corpus,
768-dim MedCPT) — DRS wins:

| Method | Recall matches undefended baseline? | Poison docs caught (of 3) |
|---|---|---|
| DRS | yes | **1** |
| L2-norm | yes | 0 |
| L2-distance | yes | 0 |
| Perplexity | no (costs recall: 0.6190/0.8078/0.8275) | 0 |

DRS ties for best on "doesn't hurt legitimate retrieval" and is the only
method that caught any poison at this scale.

**`medqa_rag` at demo scale** (`n=29` clean reference docs, Contriever
`d=768`) — DRS loses:

| Method | Detect rate |
|---|---|
| DRS | 0/3 |
| L2-norm | 1/3 |
| L2-distance | 1/3 |
| Perplexity | 0/3 |

**Why the flip:** DRS needs enough reference samples to estimate a
meaningful low-variance subspace; even after the fix removes the spurious
zero-eigenvalue directions, the *real* directions it estimates from very few
samples are still noisy (small-sample eigenvalue estimation bias — the
classic reason PCA on `n < d` data is a hard problem, not something the
dual-PCA identity alone can fully solve). 200 samples gives DRS enough
headroom in a 768-dim space to find a couple of genuinely informative
directions; 29 doesn't. Simpler statistics like L2-norm (a single global
scalar per embedding) don't need a stable multi-directional covariance
estimate at all, so they degrade far less with a tiny reference set.

## Practical recommendation

Use as large a clean reference set as the deployment reasonably supports —
closer to or above the embedding dimensionality if possible. With a small
reference set, don't trust a DRS-only evaluation; run it alongside L2-norm/
L2-distance (both now available via `--compare_defenses` in
`trial_retrieval`/`strategyqa_agent` and `--method all` in `medqa_rag`) and
compare, since at small `n` the simpler baselines may actually detect more
than DRS does.
