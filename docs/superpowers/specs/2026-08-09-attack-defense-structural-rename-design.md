# attacks/ + defenses/ + subproject rename — design

## Context

This spec is a course-correction on the original
`2026-08-08-safematch_v3-infra-attack-defense-refactor-design.md`. At the
very start of that design's brainstorming, the user was asked whether the
three subprojects should keep their separate top-level directories (with
shared code extracted into `infra/`) or be collapsed into function-based
packages (`infra/`, `attacks/`, `defenses/`), and chose **"Collapse into
attack/defense/infra packages."** When the spec was written up afterward,
its Non-goals section stated "no renaming of the `Retrieving_stage/`,
`RAG_Setting/`, `Agent_Setting/` top-level directories" — which actually
implemented the *other*, un-chosen option, without flagging the drift to
the user. Phases 1–6 were executed under that drifted plan: `infra/` grew
into a real shared package (`rag_infra.llm`, `rag_infra.data`,
`rag_infra.defenses.l2_norm`), but no top-level `attacks/`/`defenses/`
packages exist, and the three subprojects were never renamed.

The user caught this mid-Phase-7 and asked for it to be corrected. This
spec covers the correction: physically splitting attack and defense code
out of the three subprojects into new top-level `attacks/` and `defenses/`
packages, and renaming the three subprojects to reflect what they run
rather than carrying no functional signal in their names.

## Decisions made during this brainstorming session

- `rag_infra.defenses.l2_norm` (built in Phase 5) moves into the new
  `defenses/` package. `infra/` becomes purely non-attack/non-defense
  infrastructure (LLM clients, dataset file-I/O) after this move.
- `drs_defense/` stays exactly where it is — not physically relocated
  under `defenses/`. It's a working, independently-versioned package
  already embedded via `-e ../drs_defense` in all three conda
  environments; moving it would mean updating and reinstalling in all
  three for zero functional benefit, since the *package boundary*, not
  its filesystem location, is what matters for defense-centered
  organization. `defenses/` and `drs_defense/` become sibling shared
  packages, both consumed by the (renamed) subprojects.
- The three subprojects get renamed:
  - `Retrieving_stage/` → `trial_retrieval/`
  - `RAG_Setting/` → `medqa_rag/`
  - `Agent_Setting/` → `strategyqa_agent/`
- `RAG_Setting`'s attack code: only `PoisonedRAGBlackBoxGenerator` (the
  class itself) and the two attack-only prompt functions
  (`poison_generation_system_prompt`, `poison_generation_user_prompt`,
  currently in `llm/prompts.py`) move into `attacks/`. `QAItem` and
  `PoisonDoc` — investigated during this brainstorming session and found
  to be used well beyond the attack generator (by `evaluation/rag_eval.py`,
  `data/medqa_loader.py` which *produces* `QAItem`, and the
  `run_drs.py`/`run_defense.py`/`eval_attack.py` scripts) — stay in
  `medqa_rag/`'s `datamodels.py`, the same as `CorpusDoc` already does.
  `attacks/` imports them back from `medqa_rag/`, an accepted dependency
  direction since they're genuinely `medqa_rag`-owned domain types that
  predate and outlive any single attack. This also means Phase 2's
  decision to leave `medqa_loader.py` in place stands unchanged — this
  spec doesn't reopen it.
- `Retrieving_stage`'s (→ `trial_retrieval`'s) poison-generation code
  (`generate_poison_trials`, `build_poison_text`, `corpus_entry_to_example`,
  etc., currently inline in `run_poisonrag_experiment.py`) also moves into
  `attacks/`, as a second, separate module — per Phase 4's finding, these
  two attacks are different algorithms, not duplicates, so they don't get
  merged, just relocated to live under the same top-level package (mirroring
  how `rag_infra.llm` already holds three distinct call-shapes without
  merging them).

## Goal

Reach the target layout the very first brainstorming session sketched,
now informed by everything learned executing Phases 1–6:

```
repo root/
  infra/                          # rag_infra: LLM clients + dataset file-I/O only
    src/rag_infra/
      llm/                        # unchanged from phases 1
      data/                       # unchanged from phase 2
  drs_defense/                    # unchanged, stays where it is
  attacks/                        # NEW top-level package
    src/rag_attacks/
      poisonedrag_medqa.py        # PoisonedRAGBlackBoxGenerator + its 2 prompts (from medqa_rag)
      poisonedrag_trial.py        # generate_poison_trials & friends (from trial_retrieval)
  defenses/                       # NEW top-level package
    src/rag_defenses/
      l2_norm.py                  # relocated from rag_infra.defenses.l2_norm
      l2_distance.py              # relocated from medqa_rag's defense/l2_distance.py
      perplexity.py               # relocated from medqa_rag's defense/perplexity.py
      common.py                   # relocated from medqa_rag's defense/common.py (BaseDetector)
      defense_baselines.py        # relocated from strategyqa_agent's ReAct/defense_baselines.py
                                   # (minus l2_norm_scores, already merged into l2_norm.py's adapter story)

  trial_retrieval/                # was Retrieving_stage/
  medqa_rag/                      # was RAG_Setting/
  strategyqa_agent/               # was Agent_Setting/
    # each keeps its own thin adapters/experiment-driver code,
    # now importing from attacks/ and defenses/ in addition to infra/ and drs_defense/
```

## Phase breakdown (each gets its own spec-informed plan → SDD execution → review)

1. **Phase 7a — `attacks/` package.** New top-level `rag-attacks` package.
   Relocate `medqa_rag`'s `PoisonedRAGBlackBoxGenerator` + its two
   attack-only prompts; relocate `trial_retrieval`'s poison-generation
   functions as a second, separate module. Both subprojects' existing
   attack call sites become thin adapters/re-exports, following the
   established Phase-1-style pattern.

   **Outcome (implemented):** both attacks relocated byte-identical
   (verified against the still-untouched originals during task review),
   kept genuinely separate (no shared helpers, no cross-imports between
   the two modules), correct dependency direction confirmed (`attacks/`
   imports `rag_infra.llm.*` directly, never through either subproject's
   re-export). `attacks/tests/` needed a `conftest.py` putting
   `Retrieving_stage/` on `sys.path` (documented in `attacks/README.md`'s
   Tests section) since `poisonrag_experiment` isn't pip-installed
   anywhere in this repo. Two notes for later phases:
   - **For phase 7 (dead-code sweep):** `Retrieving_stage`'s
     `poisonrag_experiment/run_poisonrag_experiment.py` now has two kinds
     of "unused within this file" imports that look identical to a
     linter but aren't: `import random` and
     `from poisonrag_experiment.ollama_utils import generate_json` are
     genuinely dead (their only consumers moved to `attacks/`); but
     `build_poison_text`/`corpus_entry_to_example` are *also*
     "unused-within-file" yet are intentionally kept as re-exports for
     `Retrieving_stage/tests/test_poisonedrag_trial_parity.py`'s identity
     check. There's no `__all__` distinguishing the two — the sweep needs
     to check test/import usage, not just in-file references, before
     deleting anything here.
   - **For phase 7d (doc pass):** `poisonedrag_medqa.py` imports
     `normalize_ws` from `medrag_repro.utils.text` alongside
     `QAItem`/`PoisonDoc` — unlike those two, it's a generic two-line
     whitespace helper, not really a `medqa_rag`-owned domain type. A
     candidate for a future `rag_infra.text` extraction rather than a
     permanent subproject back-dependency, if that's ever worth doing.

   See `docs/superpowers/plans/2026-08-09-attacks-package-extraction.md`.

2. **Phase 7b — `defenses/` package.** New top-level `rag-defenses`
   package. Move `rag_infra.defenses.l2_norm` here (with `infra/`'s
   dependents updated). Relocate `medqa_rag`'s `l2_distance.py`,
   `perplexity.py`, `common.py`; relocate `strategyqa_agent`'s
   `defense_baselines.py` contents (minus `l2_norm_scores`, already
   covered by the `l2_norm.py` move). `drs_defense/` and its three
   existing adapters (`medqa_rag/.../defense/drs.py`,
   `strategyqa_agent/ReAct/drs.py`, `trial_retrieval/.../drs.py`) are
   untouched — they already delegate to `drs_defense/`, which isn't
   moving.

   **Outcome (implemented):** per the mid-session decision to relocate
   whole Detector classes (not just math), `defenses/` ended up holding
   five modules: `common.py` (`BaseDetector`), `l2_norm.py`
   (`l2_norm_score` + `L2NormDetector` + `l2_norm_scores`, all three
   merged), `l2_distance.py` (`L2DistanceDetector` centroid-distance +
   `l2_distance_scores`/`leave_one_out_l2_distance_scores`
   nearest-neighbor-distance — confirmed still two different formulas,
   kept separate), `perplexity.py` (`PerplexityDetector` +
   `PerplexityScorer` — same core computation, kept as two distinct
   classes, not proven interchangeable), `defense_baselines.py`
   (`QuantileStats`/`PerplexityStats` fitting). `medqa_rag`'s (`RAG_Setting`'s)
   `defense/drs.py` needed exactly one line changed (`BaseDetector`
   import repointed) since `DRSDetector` extends it — `DRSDetector`'s own
   logic and `drs_defense/` itself stayed untouched, confirming the
   "drs_defense isn't moving" decision held even under the "move whole
   classes" approach.

   **Process lesson, worth reading before writing future phase plans:**
   this phase's implementation plan said "merge" `rag_infra.defenses.l2_norm`
   into `defenses/l2_norm.py` where this spec said "move" — the plan was
   faithfully implemented, but "merge" (copy semantics, nothing deleted)
   quietly dropped the spec's explicit instruction that `infra/` "becomes
   purely non-attack/non-defense infrastructure after this move." The
   result: `l2_norm_score` existed in two shared packages simultaneously,
   and three doc surfaces made false ownership claims — invisible to
   every per-task review, since the deletion appeared in no task's file
   list, and only surfaced in the final whole-branch review reading the
   plan against the spec, not just the diff against the plan. Fixed in a
   follow-up commit (deleted the orphaned `infra/` copy + its test,
   trimmed the three stale doc surfaces, repointed two tests that still
   imported from the deleted location). When a plan's wording drifts from
   "move" to "merge"/"consolidate," check whether the spec actually meant
   deletion of the source before writing the plan's file list.

   See `docs/superpowers/plans/2026-08-09-defenses-package-extraction.md`.

3. **Phase 7c — rename the three subprojects.** `Retrieving_stage/` →
   `trial_retrieval/`, `RAG_Setting/` → `medqa_rag/`, `Agent_Setting/` →
   `strategyqa_agent/`. Mechanical repo-wide reference update: every
   `-e ../X` relative install path stays correct automatically (sibling
   references, unaffected by the renamed directory's own name), but every
   doc, hardcoded path (e.g. `strategyqa_agent/ReAct/local_wikienv.py`'s
   hardcoded `"ReAct/database/..."` literals, which are relative to CWD at
   runtime and unaffected by the parent rename, but should be
   double-checked), and cross-reference needs updating.

   **Outcome (implemented) — scope grew mid-execution.** The rename
   itself landed exactly as planned (directories only, zero Python
   imports touched, `medrag-repro`'s pip editable install reinstalled
   from its new location since `pip install -e` bakes in an absolute
   path). Then, **after the rename finished, the user asked to also nest
   all three renamed directories under a new top-level `use-cases/`
   directory** (`use-cases/medqa_rag/`, `use-cases/strategyqa_agent/`,
   `use-cases/trial_retrieval/`), separating them from the four
   shared-library packages (`drs_defense/`, `infra/`, `attacks/`,
   `defenses/`), which stay at the repo root — this is not something
   this spec anticipated, so the target layout below supersedes the
   flat-at-root layout shown earlier in this document. The nesting
   invalidated this bullet's "every `-e ../X` path stays correct
   automatically" claim: it's true for a same-depth rename, but nesting
   one level deeper broke ten `-e ../X` lines (now `-e ../../X`) across
   three dependency manifests, plus `attacks/tests/conftest.py`'s
   hardcoded path depth, plus a second `medrag-repro` reinstall.

   **Corrected target layout** (supersedes the tree in this spec's
   "Goal" section above):
   ```
   repo root/
     infra/  drs_defense/  attacks/  defenses/     # shared libraries, unchanged
     use-cases/
       medqa_rag/           # was RAG_Setting/
       strategyqa_agent/    # was Agent_Setting/
       trial_retrieval/     # was Retrieving_stage/
   ```

   **Process lesson:** the nine-doc-file substitution and every
   verification grep in this phase searched for the three OLD DIRECTORY
   NAMES — which correctly caught every reference to them, but structurally
   cannot catch depth-relative staleness in a path that never contained
   an old name to begin with. `drs_defense/README.md` had exactly this:
   `pip install -e ../drs_defense`, correct before the nesting, silently
   wrong after it, invisible to every grep in this plan, caught only in
   final review. When a future phase changes nesting depth (not just
   names), also grep documentation and manifests for `../` path literals,
   not just renamed tokens.

   **Follow-up surfaced, not yet actioned:** 30 tracked `.pyc` files and
   5 tracked `.DS_Store` files exist repo-wide (already `.gitignore`d,
   but gitignore doesn't retroactively untrack already-tracked files).
   This caused two separate fix rounds within this phase alone (a
   false "restored" claim on 3 `trial_retrieval` `.pyc` files, then a
   dirty working tree from the same files regenerating again). The
   `drs-shared-module` branch already fixed this once (commit `0052ac1`,
   pre-dating this branch's history) — this branch never got that fix.
   A `git rm --cached` untracking sweep of all 35 files would retire the
   failure mode permanently; recommended as part of the original Phase 7
   dead-code sweep, since it's the same class of repo-hygiene cleanup.

   See `docs/superpowers/plans/2026-08-10-rename-subprojects.md`.

4. **Phase 7d — documentation pass.** `CLAUDE.md`, root `README.md`, and
   each subproject's `README.md` rewritten to describe the final
   structure coherently in one pass, rather than patched incrementally
   the way the last six phases' final reviews caught doc gaps one at a
   time.

   **Outcome (implemented).** Rewrote all 10 doc files — root `README.md`,
   `CLAUDE.md`, the 4 shared-library `README.md`s (`drs_defense/`,
   `infra/`, `attacks/`, `defenses/`), and the 3 `use-cases/` subproject
   `README.md`s including the nested `poisonrag_experiment/README.md` —
   each now covering code structure, functional modules, an install
   guide, quick-start steps, and a runnable example, per the six-task
   plan at `docs/superpowers/plans/2026-08-10-phase-7d-documentation-pass.md`.
   Two real pre-existing bugs surfaced and were fixed, not just
   rewritten around: `use-cases/trial_retrieval/README.md` and its
   nested `poisonrag_experiment/README.md` linked to another user's
   local machine path (`/Users/ningzeqiang/Downloads/TrialGPT-main/...`)
   instead of relative in-repo paths, and the latter's "Run" section said
   "From repo root:" when the command actually requires running from
   `use-cases/trial_retrieval/`. Also documented, not silently
   "corrected": `use-cases/trial_retrieval/`'s own scripts
   (`keyword_generation.py`'s `DEFAULT_MODEL`,
   `poisonrag_experiment/run_poisonrag_experiment.py`'s `--ollama_model`
   default) literally default to the Ollama tag `qwen-2.5:7b-instruct`
   (hyphenated) — a tag Ollama doesn't publish, distinct from the
   `qwen2.5:7b-instruct` tag used everywhere else in this repo. This is a
   pre-existing code inconsistency out of scope for a docs-only phase;
   every runnable example in that subproject's docs now passes
   `qwen2.5:7b-instruct` explicitly rather than relying on the broken
   default.

   **Process lesson:** Task 5 (the `medqa_rag/README.md` rewrite)
   embedded a YAML config block sourced, during planning, from the *old*
   README's own copy of it rather than the live
   `configs/minimal_medqaus_pubmed_contriever.yaml` file — the two had
   already drifted (`batch_size: 8` vs. live `32`, `max_trials: 50` vs.
   live `15   #50`). The Task 5 implementer caught this via the plan's
   own verification step (diff the embedded block against the live file)
   and correctly reported `DONE_WITH_CONCERNS` instead of silently
   reproducing or silently fixing the stale content. Fixed in two
   follow-up commits (the README and the plan document itself). Lesson,
   generalizing the Phase 7c one about `../` path literals: when a plan
   embeds a copy of a config/data file for a *rewrite* task, source it
   from the live file at plan-authoring time, not from the file being
   replaced — the file being replaced is exactly the thing already
   suspected of being stale.

   The final whole-branch review found 0 Critical, 2 Important (both
   fixed: `attacks/README.md`/`defenses/README.md` each had a per-module
   bullet list left without a heading once new sections were inserted
   above it by an earlier task — a cross-task consistency gap no
   single-task review could see; two `use-cases/` READMEs never stated
   their Install/Quick-start commands must run from the subproject
   directory, unlike their sibling `poisonrag_experiment/README.md`
   which got that exact fix in the same phase) and 7 Minor findings,
   all deferred as non-load-bearing (cosmetic tree-comment alignment,
   `drs_defense/README.md`'s differently-ordered sections as prescribed
   by the plan, missing `## Tests` sections in `use-cases/` READMEs since
   Tests wasn't part of this phase's five-part scope, and similar
   polish). See
   `docs/superpowers/plans/2026-08-10-phase-7d-documentation-pass.md`.

5. **Original Phase 7 — dead-code sweep.** Runs last, after the structure
   settles, per the original spec (`vulture`-based scan, manual triage).
   The findings already gathered from Phase 6's final review (safe/unsafe
   `environment.yml` packages, the vestigial `agentpoison` env name)
   carry forward into this pass.

## Non-goals

- No merging of the two attack implementations (`poisonedrag_medqa.py`,
  `poisonedrag_trial.py`) — confirmed in Phase 4 to be different
  algorithms, staying separate.
- No merging of `l2_distance`'s two different formulas, or extraction of
  `perplexity`'s torch/transformers-dependent core math — both still
  deferred per Phases 5's findings; this restructuring only *relocates*
  what already exists, it doesn't resolve those deferred consolidation
  questions.
- No physical relocation of `drs_defense/`.
- No changes to `infra/`'s `llm`/`data` subpackages beyond removing the
  now-relocated `defenses` subpackage.
- No renaming of `infra/`'s package/dist name (`rag_infra`/`rag-infra`)
  — it stays `infra/` at the repo root, just with narrower contents.

## Process note

Given the size (new packages, a repo-wide directory rename, a full doc
rewrite), each of the four new phases (7a–7d) gets its own implementation
plan via `writing-plans`, executed and reviewed independently — same
process as Phases 1–6. Phase 7c (the rename) should run after 7a/7b so
the new `attacks/`/`defenses/` packages' internal paths are stable before
every reference to the subprojects' directory names gets rewritten in one
pass; running the rename first would mean touching those references
twice.
