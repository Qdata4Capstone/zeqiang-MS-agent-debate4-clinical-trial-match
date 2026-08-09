# safematch_v3 infra/attack/defense refactor — design

## Context

This spec supersedes the scope of the earlier
`2026-08-08-safematch_v3-dead-code-cleanup-design.md` (dead-code cleanup
only). The user asked for a larger structural refactor: extract the
infrastructure that every experiment (attack or defense) runs on top of —
retrieval/embedding, LLM clients, dataset loaders — into shared modules
(the same pattern already used for DRS math in `drs_defense/`), then trim
each subproject down to only RAG-defense code plus one simple, canonical
RAG attack. The dead-code `vulture` sweep from the earlier spec is folded
into this plan as its final phase, run after relocation so it also catches
anything orphaned by the moves.

## Current state (established by prior analysis)

`safematch_v3` (forked from `drs-shared-module`'s tip) contains three
independent subprojects, each with its own environment:

- `Retrieving_stage/` — plain pip, TrialGPT-style retrieval (BM25 +
  MedCPT hybrid) plus `poisonrag_experiment/` (corpus-poisoning attack +
  DRS defense, self-contained poison generation inline in
  `run_poisonrag_experiment.py`).
- `RAG_Setting/` — `medrag` conda env, Python 3.10. PoisonedRAG black-box
  attack (`attacks/poisonedrag_blackbox.py`) against a Contriever-based
  medical QA RAG pipeline, with DRS/`l2_norm`/`l2_distance`/`perplexity`
  defenses.
- `Agent_Setting/` — `agentpoison` conda env, Python 3.9, CUDA-pinned
  `torch==2.0.1`. Two independent pieces: `algo/trigger_optimization.py`
  (gradient-guided adversarial trigger optimization against a dense
  retriever — a materially different, more complex attack class) and
  `ReAct/` (a ReAct agent over StrategyQA, using DPR retrieval, with DRS
  and `defense_baselines.py` quantile/perplexity defenses).
- `drs_defense/` — already-shared DRS reference implementation, consumed
  by thin adapters in all three subprojects. This extraction is the proven
  precedent for the rest of this refactor: shared module + thin adapter +
  parity test proving the adapter matches the shared implementation exactly.

## Goal

Restructure the repo around function rather than experiment:

- One shared `infra/` package (retrieval/embedding backends, LLM clients,
  dataset loaders, the ReAct agent-environment harness) that every
  experiment imports instead of duplicating.
- One shared `attacks/` package containing a single canonical
  PoisonedRAG-style black-box poisoning attack, reused by both
  `RAG_Setting` and `Retrieving_stage` in place of their two separate
  implementations.
- One shared `defenses/` package containing `drs_defense/` (unchanged),
  `l2_norm.py`, `l2_distance.py`, `perplexity.py`, and
  `defense_baselines.py` (merged into the above if its math overlaps).
- `Agent_Setting/algo/` (`trigger_optimization.py`, `utils.py`,
  `config.py`) deleted entirely — out of scope as "not RAG defense, not a
  simple attack."
- `Retrieving_stage/`, `RAG_Setting/`, `Agent_Setting/` remain as thin
  experiment drivers (config, CLI entry points, glue) importing from
  `infra/`, `attacks/`, `defenses/` instead of containing the logic
  themselves.

## Target layout

```
repo root/
  infra/
    retrieval/
      bm25_medcpt.py            # from Retrieving_stage/trialgpt_retrieval
      contriever.py             # from RAG_Setting/src/medrag_repro/retriever
      dpr.py                    # DPR ctx-encoder retrieval, used by ReAct's --model dpr search
    llm/
      client.py                 # merged OpenAI-compatible client (RAG_Setting's client.py + Agent_Setting's ollama_client.py)
      prompts.py
    data/
      medqa_pubmed.py
      sigir_trec.py
      strategyqa.py
    agent_env/                  # ReAct harness, kept as the defense-evaluation environment
      local_wikienv.py
      wrappers.py
      search.py

  attacks/
    poisonedrag.py              # ONE canonical black-box poisoning attack

  defenses/
    drs_defense/                # existing shared module, untouched
    l2_norm.py
    l2_distance.py
    perplexity.py
    defense_baselines.py        # merged in if math overlaps with l2_norm/perplexity

  Retrieving_stage/  RAG_Setting/  Agent_Setting/
    # thin experiment drivers only: config, CLI entry points, glue code
```

## Migration phases (lowest risk first)

Each phase follows the `drs_defense` precedent: add the shared
implementation, add a parity test proving it matches the original(s)
exactly, switch the subproject(s) to the shared version via a thin
adapter, run the full test surface, only then delete the original.

1. **`infra/llm/`** — merge `RAG_Setting`'s OpenAI-compatible client and
   `Agent_Setting`'s `ollama_client.py`. No CUDA/version-pinning risk;
   both are thin HTTP wrappers against an OpenAI-compatible endpoint.

2. **`infra/data/`** — relocate dataset loaders (MedQA/PubMed, SIGIR/TREC,
   StrategyQA) with straight code motion and updated import paths.

   **Outcome (implemented, narrowed):** research before writing the phase 2
   plan found this assumption didn't hold uniformly. `RAG_Setting`'s
   `medqa_loader.py`/`pubmed_loader.py` are coupled to subproject-specific
   dataclasses (`QAItem`, `CorpusDoc`) and `Agent_Setting/ReAct`'s
   StrategyQA loading is a method on the stateful `WikiEnv` class — neither
   is a clean standalone relocation. Only `Retrieving_stage`'s four
   dependency-free file-I/O helpers (`load_jsonl`, `dump_json`, `load_qrels`,
   `load_queries_and_keywords`) were extracted, as `rag_infra.data.jsonl`.
   See `docs/superpowers/plans/2026-08-09-infra-data-jsonl-extraction.md`.

3. **`infra/retrieval/`** — relocate BM25/MedCPT, Contriever, and DPR
   backends as separate files (relocated, not merged into one algorithm).
   Highest-risk step: `Agent_Setting` pins Python 3.9 + CUDA-specific
   `torch==2.0.1` while `RAG_Setting` uses Python 3.10, so the shared
   package's code must stay syntax-compatible with both; each subproject
   still installs it into its own conda env (no environment unification).

   **Outcome (skipped, on hold):** research found `RAG_Setting`'s
   `ContrieverEncoder`/`retriever/index.py` and `Retrieving_stage`'s
   BM25/MedCPT code in `retrieval_utils.py` are both genuinely standalone
   and relocatable, but `Agent_Setting`'s DPR-based retrieval is a method
   (`WikiEnv._embed_text`) entangled with the stateful ReAct environment
   (DRS fitting, defense baselines, hardcoded `.to("cuda")`) — not
   separable without restructuring that class, which is out of scope. This
   phase would also be the first to add real ML dependencies (`torch`,
   `transformers`, `faiss`, `nltk`, `rank_bm25`) to `rag_infra`, with
   `RAG_Setting` and `Retrieving_stage` currently pinning slightly
   different `torch`/`transformers` versions. Deferred rather than
   attempted narrowed, pending a decision on how to handle the dependency
   footprint (e.g. optional extras) — not started.

4. **`attacks/poisonedrag.py`** — consolidate `RAG_Setting`'s
   `poisonedrag_blackbox.py` and `Retrieving_stage`'s inline poison-gen
   logic (currently inside `run_poisonrag_experiment.py`) into one
   canonical implementation, proven via parity tests against both
   originals' outputs before either is deleted.

   **Outcome (skipped, not applicable):** research found these are not
   duplicate implementations of one technique — they're different attacks.
   `RAG_Setting`'s `PoisonedRAGBlackBoxGenerator` implements the actual
   PoisonedRAG black-box algorithm (generate candidate → verify the target
   LLM answers the target wrong MCQ option → retry up to `max_trials`),
   coupled to MCQ-specific dataclasses (`QAItem`, `PoisonDoc`).
   `Retrieving_stage`'s `generate_poison_trials` is a one-shot synthetic
   clinical-trial-record generator with no verify/retry loop at all — it
   doesn't implement the PoisonedRAG algorithm's core mechanism, despite
   the similar naming. There is no shared output ever produced to prove
   parity against, so a "canonical merge" isn't a coherent goal here; both
   implementations are left as-is.

5. **`defenses/`** — relocate `l2_norm`/`l2_distance`/`perplexity` from
   `RAG_Setting`; merge `Agent_Setting`'s `defense_baselines.py` in if its
   quantile/perplexity math is equivalent, otherwise keep it as a
   distinct file in the same package.

   **Outcome (implemented, narrowed):** only `l2_norm` was a genuine
   duplicate — `RAG_Setting`'s `np.linalg.norm(X, axis=1)` and
   `Agent_Setting`'s `torch.norm(embeddings.float(), dim=1)` compute the
   identical formula, extracted as `rag_infra.defenses.l2_norm.l2_norm_score`
   with both subprojects' detectors delegating to it (mirroring the
   `drs_defense`/`ReAct/drs.py` numpy-core + torch-adapter pattern). Kept
   separate:
   - **`l2_distance`**: `RAG_Setting`'s `L2DistanceDetector` scores
     distance-to-centroid-of-clean-embeddings; `Agent_Setting`'s
     `defense_baselines.l2_distance_scores` scores
     distance-to-nearest-individual-clean-embedding (KNN-style, via
     `torch.cdist(...).min(dim=1)`) — different statistics, not a
     duplicate, same situation as phase 4's two "PoisonedRAG" attacks.
     Note found during final review: the *actual* centroid-distance
     duplicate of `L2DistanceDetector` is
     `Agent_Setting/algo/trigger_optimization.py`'s fitness function
     (`torch.norm(embeddings - mean_embedding, dim=1)`, line ~69), not
     `defense_baselines.py` — but `algo/` is slated for full deletion in
     phase 6, so this potential consolidation is moot once that lands.
   - **`perplexity`**: `RAG_Setting`'s `PerplexityDetector` and
     `Agent_Setting`'s `PerplexityScorer` do compute the same core value
     (`exp(causal-LM loss)`), but extracting it would be the first
     `rag_infra` module needing `torch`+`transformers` as hard
     dependencies — deferred alongside phase 3 for the same reason.
   - `RAG_Setting`'s `BaseDetector` (`defense/common.py`) stays
     `RAG_Setting`-local — a generic ABC, but nothing outside
     `RAG_Setting` uses an equivalent pattern, so there's nothing to
     deduplicate. See
     `docs/superpowers/plans/2026-08-09-defenses-l2-norm-extraction.md`.

6. **Delete `Agent_Setting/algo/`** — first confirm (via grep across the
   repo, not assumption) that nothing outside `algo/` imports from it,
   in particular `ReAct/run_strategyqa_inference.py`, before deleting
   `trigger_optimization.py`, `utils.py`, and `config.py`.

   **Outcome (implemented):** confirmed zero importers repo-wide (not just
   within `Agent_Setting/`) before deleting; `CLAUDE.md` and both root and
   `Agent_Setting/README.md` updated to no longer describe the deleted
   capability (the final review caught a root-`README.md` miss the
   deletion's own path-shaped verification grep couldn't detect, since it
   was a prose description, not an import). `Agent_Setting/environment.yml`
   deliberately left untouched — see the pre-triaged inputs below for
   phase 7. See
   `docs/superpowers/plans/2026-08-09-delete-agent-setting-algo.md`.

   **Pre-triaged for phase 7** (verified during phase 6's final review):
   safe to prune from `environment.yml` — `wandb`, `autogen==1.0.16`,
   `pyautogen==0.2.0`, `wolframalpha==5.0.0`, `casadi==3.6.5`,
   `shapely==2.0.5` (zero references remain anywhere in tracked `.py`
   files). Do NOT prune `gym==0.26.2` — load-bearing for
   `ReAct/local_wikienv.py` (`WikiEnv(gym.Env)`, `textSpace(gym.spaces.Space)`)
   and all five wrapper classes in `ReAct/wrappers.py`. The `agentpoison`
   conda env name (`environment.yml`, `Agent_Setting/README.md`,
   `CLAUDE.md`) is now a vestige of the deleted code but renaming it
   breaks existing local envs — bundle with the dependency prune, don't
   do alone.

7. **Dead-code sweep** — run the `vulture`-based scan (per the prior
   cleanup spec) across the new layout, now that relocation may have
   orphaned additional code. Manually triage every hit before removal,
   using the same false-positive checklist as the prior spec (CLI entry
   points, `__init__.py` re-exports, pytest fixtures, dunder methods, ABC
   overrides, dynamic dispatch).

## Testing strategy

- Before deleting any original implementation in a phase, add a parity
  test proving the shared version's output matches it exactly (same
  pattern as `drs_defense`'s three adapter parity suites).
- After each phase: run `drs_defense/tests/` plus the (growing) set of
  parity suites, and smoke-run each affected experiment's entry-point
  script end-to-end against small/sample data — this repo has no other
  test suite or CI, so script-level smoke runs are the only correctness
  signal for non-parity-tested code (per CLAUDE.md).
- A phase is not complete until its parity tests pass AND the relevant
  entry-point script(s) still run successfully.

## Process note: one plan per phase

Given the size (three separate conda environments, CUDA-version risk in
phase 3, and a canonical-attack merge in phase 4 that needs behavioral
proof), this spec covers the *target architecture and phase ordering*
only. Each phase gets its own implementation plan via `writing-plans`,
executed and reviewed independently, rather than one monolithic plan —
so review checkpoints land between phases instead of only at the end.
Phase 1 (`infra/llm/`) is the lowest-risk starting point.

## Non-goals

- No renaming of the `Retrieving_stage/`, `RAG_Setting/`, `Agent_Setting/`
  top-level directories — they remain as experiment-driver homes; only
  the reusable engine code moves out of them.
- No unification of the three conda environments — `infra/` code must
  stay compatible with all of them, but each subproject keeps installing
  it into its own env.
- No changes to `drs_defense/`'s core math or its existing adapters'
  behavior.
- No new attack techniques or defense methods — this is a relocation and
  consolidation of what already exists, not new research code.

## Risks

- **CUDA/version drift (phase 3):** `infra/retrieval/dpr.py` must work
  under both `RAG_Setting`'s Python 3.10 env and `Agent_Setting`'s
  Python 3.9 + `torch==2.0.1` env. Mitigated by keeping backend files
  separate per retriever rather than sharing code paths across
  incompatible dependency versions.
- **Silent behavior drift in the attack merge (phase 4):** the two
  PoisonedRAG-style implementations may have already diverged (similar to
  how the four DRS reimplementations had drifted before the
  `drs_defense` extraction). Mitigated by parity tests against both
  originals before deletion, not just against one.
- **Hidden import of `Agent_Setting/algo/` (phase 6):** deleting it
  without a repo-wide grep first could silently break `ReAct/` if
  anything there imports shared helpers from `algo/utils.py`.
- **Scope creep across 7 phases:** mitigated by the one-plan-per-phase
  process above, so each phase can be reviewed and merged (or halted)
  independently.
