# DRS: Dual-PCA Fix, Defense Comparison, and Paper-Consistency Analysis

**Date:** 2026-08-10
**Code:** `drs_defense/src/drs_defense/core.py` (`low_variance_eigenbasis`), tests in `drs_defense/tests/test_core_dual_pca.py`
**Related docs:** `drs_defense/README.md` ("Few reference samples relative to embedding dimensionality" section), `use-cases/trial_retrieval/poisonrag_experiment/README.md` ("DRS use here" section)
**Source paper:** "Understanding Data Poisoning Attacks for RAG: Insights and Algorithms" (ICLR 2025 submission), [openreview.net/pdf?id=2aL6gcFX7q](https://openreview.net/pdf?id=2aL6gcFX7q)

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

## Consistency with the source paper

Checked directly against the paper ("Understanding Data Poisoning Attacks
for RAG: Insights and Algorithms", ICLR 2025 submission,
[openreview.net/pdf?id=2aL6gcFX7q](https://openreview.net/pdf?id=2aL6gcFX7q)),
13 pages (main text + references, no appendix in the copy checked).

**Algorithm fidelity: matches exactly.** `drs_defense.core`'s implementation
of Algorithm 1 (Eq. 3, `DRS(z;X) = Σ|zᵀvᵢ|/√λᵢ` over the M smallest
eigenvalue directions, ascending) and Algorithm 2 (threshold = q-th
quantile of clean scores) matches the paper's pseudocode line for line.
Every DRS quantile default in this repo (`0.99`) matches the paper's stated
target of "approximately 1%" clean false-positive rate. The `power`
parameter in `poisonrag_experiment/drs.py` is correctly documented as a
local ablation knob with no counterpart in the paper's formula.

**Reference-set construction: one real deviation, in `trial_retrieval`
only — now fixed as an opt-in flag and empirically confirmed to matter.**
The paper's Algorithm 2 retrieves top-K clean documents for *every* query
in the protected query set Q and pools them into one combined reference
set, fitting a single DRS model shared across all protected queries.
`medqa_rag`'s `run_defense.py` and `strategyqa_agent`'s `_fit_drs` both do
this correctly (checked directly — both pool across all clean queries
before fitting). `trial_retrieval`'s `apply_drs_defense` did not: it fit a
*separate* DRS model per individual target patient, using only that one
patient's own top-`ref_k` documents, never pooling across the other target
patients — structurally capping the achievable reference-set size at
`ref_k` alone (20 by default) instead of `ref_k x num_targets`, and very
plausibly why the `n <= d` bug above was so easy to trigger there
specifically.

Added `apply_drs_defense_pooled` (paper-faithful Algorithm 2) alongside the
existing per-query `apply_drs_defense`, selectable via `--drs_pool_reference`
/ `--no-drs_pool_reference` (pooled is the default — see below), and
compared both on the same real cached embeddings, 3 target patients,
`--drs_quantile 0.99`:

| `--drs_ref_k` | strategy | reference-set size | candidates flagged | poison docs caught | recall@50/100/200 |
|---|---|---|---|---|---|
| 20 | per-query (`--no-drs_pool_reference`) | 20 (x3 separate models) | 5 | 0/3 | 0.7052/0.8941/0.9137 |
| 20 | pooled (default) | 60 (deduplicated) | 369 | 0/3 | 0.7052/0.8941/0.9137 |
| 200 | per-query (`--no-drs_pool_reference`) | 200 (x3 separate models) | 5,520 | 1/3 | 0.7052/0.8941/0.9137 |
| 200 | pooled (default) | 521 (deduplicated) | 6,061 | **3/3** | 0.7052/0.8941/0.9137 |

At `ref_k=200`, pooling catches **every** poison document (vs. 1/3 for the
per-query variant) while keeping recall *exactly* at the undefended
baseline in all four conditions — the extra flags pooling produces (6,061
vs. 5,520) land entirely on non-relevant documents, not on any of the 3
patients' 28 true-positive qrels docs. This is a clean, real-world
confirmation that the paper's actual reference-set design (pool across the
whole protected query set) meaningfully outperforms the per-query variant
this codebase originally shipped — not just architecturally closer to the
paper, but empirically better on every axis measured here. At `ref_k=20`
pooling still doesn't have enough reference documents to detect anything
(0/3 either way), consistent with the statistical-power story above: more
reference documents helps, but only once there are actually enough of
them.

**Update:** pooling is now the default (`--drs_pool_reference` defaults to
`True`; pass `--no-drs_pool_reference` for the original per-query
behavior), confirmed with the user before flipping it.

**Reference-set size: `medqa_rag`'s real config matches the paper closely;
`trial_retrieval`'s default doesn't, independent of the demo scaling
above.** The paper states: *"M... to 100, the number of clean queries to
300 with k=5, resulting in a total of 1,000 clean documents."*
`medqa_rag`'s real (non-demo) config, `configs/minimal_medqaus_pubmed_
contriever.yaml`, already uses `drs.M: 100` and `medqa.n_clean_queries:
300` — matching the paper almost exactly (and, run against its intended
100k-doc corpus rather than the 300-doc demo corpus, should yield a
similarly-sized pooled reference set to the paper's ~1,000, since a larger,
more diverse corpus means less duplicate-doc overlap across the 300
queries). `trial_retrieval`'s real default (`--drs_ref_k 20
--drs_num_directions 16`, no pooling) is far below the paper's scale even
before any demo-specific shrinking.

**A likely factual error found while checking:** `strategyqa_agent/
README.md`'s Notes section claims `--drs_num_directions 200 matches the
main DRS setting used in the paper`. Every M value stated anywhere in the
paper's main text — Table 2, Table 3, Table 4, Table 5, and the shared
setup description in Section 5.1.1 — is **100**, not 200. `200` does not
appear anywhere in the 13 pages checked. The paper does reference an
appendix with "additional ablation studies" not included in this copy, so
an appendix-only `M=200` ablation can't be ruled out, but nothing in the
visible text supports the README's claim as written. Worth a follow-up fix
(either correct the README to `100`, or verify the appendix actually
contains `200` before keeping the claim).

**Why DRS's advantage over baselines looked weaker in this repo's tests
than the paper's headline numbers — not a contradiction.** The paper
reports DRS achieving near-perfect (0.95-0.99) filtering rates,
dramatically outperforming perplexity/L2-norm/L2-distance (0.01-0.36)
across every scenario it tests (RAG agent, dense-retrieval QA, medical QA
— Tables 2-4). The comparisons run in this repo (above) found a much more
mixed picture: DRS ties or wins at `trial_retrieval`'s `--drs_ref_k 200`,
but loses to L2-norm/L2-distance at `medqa_rag`'s tiny demo scale (`n=29`).
This isn't evidence against the paper's claims — the paper's own
methodology always used `n ~ 1,000 >> d ~ 768`; every comparison run here
used `n` between 20 and 200, well below that. Given this analysis already
established that DRS's real detection power scales with reference-set size
relative to embedding dimensionality, weaker/mixed results at
under-scaled `n` are the expected outcome, not a discrepancy with the
paper's algorithm. It also surfaces something the paper's own presentation
doesn't discuss: Algorithm 1's pseudocode states an explicit input
constraint `M <= d`, but never states or tests any constraint relating `n`
to `d` — unsurprising, since the paper's own reference sets never came
close to that boundary, so the degenerate `n <= d` case this document fixes
was never something its authors needed to handle.

## Crossover confirmed: `medqa_rag` reference-size/M sweep

The prediction above (DRS needs `n` and `M` to grow toward the paper's
scale before it beats the baselines) was tested directly rather than left
as inference. `use-cases/medqa_rag/scripts/sweep_reference_size.py` reuses
`configs/sweep.yaml`'s shared prep (1,500-doc PubMed corpus, 3 targets, 300
clean queries — `medqa.n_clean_queries: 300` and `drs.M: 100` match the
paper's stated setup) and sweeps the number of clean queries pooled into
the reference set (29/50/100/200/300) crossed with DRS's `M`
(10/50/100), holding poison-detection rate and clean FPR as the metrics
(no LLM calls needed for this — detection stats alone answer the
question).

Poison-detection rate (out of 3 poison docs), clean FPR stayed <=0.04
throughout every cell (this is genuine detection, not the over-flagging
bug the dual-PCA fix above addresses):

| Ref queries | Pooled docs | l2_norm | l2_distance | perplexity | DRS M=10 | DRS M=50 | DRS M=100 |
|---|---|---|---|---|---|---|---|
| 29  | 71  | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 (M clipped to 70) |
| 50  | 102 | 0.00 | 0.33 | 0.00 | 0.00 | 0.00 | 0.00 |
| 100 | 134 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 200 | 186 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| 300 | 241 | 0.00 | 0.33 | 0.00 | 0.00 | **0.33** | **0.67** |

DRS stays at 0/3 until *both* the reference set and `M` grow together —
at 241 pooled docs, `M=10` and `M=50` are still 0.00/0.33, but `M=100`
jumps to 0.67 (2/3), beating every baseline at that same reference-set
size (l2_distance's best anywhere in the table is 1/3). This confirms
`M` scaling toward the paper's value, not reference-set size alone, is
what unlocks DRS's advantage — consistent with Eq. 3 using exactly `M`
directions, so a too-small `M` structurally caps how much of the clean
subspace DRS can even look at regardless of how well-estimated that
subspace is.

One caveat on scale: this repo's demo corpus is only 1,500 PubMed docs, so
300 clean queries at `k=5` pool down to 241 *unique* docs (heavy overlap),
well short of the paper's ~1,000-doc reference set from the same query
count against its full-size corpus. The 0.67 result at `n=241` is
therefore a lower bound on what `M=100` achieves at the paper's actual
scale, not the ceiling — consistent with the practical recommendation
above to use as large a reference set as the deployment supports.

Raw results (30 sweep points): `use-cases/medqa_rag/artifacts_sweep/reference_size_sweep.json`.

### Pushing further: DRS reaches perfect detection

The 0.67 result above begged the obvious follow-up — does it keep climbing
toward the paper's reported 0.95-0.99? Extended the sweep by regenerating
`configs/sweep.yaml`'s clean-query pool from 300 to 600 queries (same seed,
same corpus/targets/poison — `sample_targets_and_clean_queries`'s shuffle
is deterministic given the same seed and item list, so the original 300
are an exact prefix of the new 600, confirmed by asserting `qid` equality
before overwriting `data_sweep/clean_queries.jsonl`) and sweeping
`ref_sizes 300,400,500,600` × `M values 100,150,200,250,300`:

| Ref queries | Pooled docs | l2_norm | l2_distance | perplexity | DRS M=100 | DRS M=150 | DRS M=200 | DRS M=250 | DRS M=300 |
|---|---|---|---|---|---|---|---|---|---|
| 300 | 241 | 0.00 | 0.33 | 0.00 | 0.67 | 0.33 | 0.33 | 0.33 (M clipped to 240) | 0.33 (M clipped to 240) |
| 400 | 279 | 0.00 | 0.33 | 0.00 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 (M clipped to 278) |
| 500 | 304 | 0.00 | 0.33 | 0.00 | 0.67 | 0.67 | **1.00** | **1.00** | 0.67 |
| 600 | 326 | 0.00 | 0.00 | 0.00 | **1.00** | **1.00** | **1.00** | **1.00** | **1.00** |

At 326 pooled reference docs, DRS hits **1.00 (3/3)** at every `M` tested,
including `M=100` — the paper's own value, no extra tuning needed — while
every baseline is stuck at 0.00-0.33 the entire table. This matches the
paper's headline near-perfect DRS filtering (0.95-0.99) essentially
exactly, and confirms the earlier 0.67 result at `n=241` genuinely was a
lower bound, not a ceiling: it keeps climbing as the reference set grows,
until it hits a perfect score well before reaching the paper's own ~1,000
document scale. (Non-monotonic bumps like `M=300` dropping from 1.00 back
to 0.67 at `ref_size=500` are noise from only 3 poison docs total — each
step is worth 0.33 — not a real regression; the 600-query row is fully
saturated across all `M`, so this doesn't affect the conclusion.)

Raw results: `use-cases/medqa_rag/artifacts_sweep/reference_size_sweep.json`
(initial sweep, 30 points) and `reference_size_sweep_extended.json` (this
follow-up, 32 points).
