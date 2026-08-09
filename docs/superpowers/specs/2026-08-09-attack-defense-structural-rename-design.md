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

3. **Phase 7c — rename the three subprojects.** `Retrieving_stage/` →
   `trial_retrieval/`, `RAG_Setting/` → `medqa_rag/`, `Agent_Setting/` →
   `strategyqa_agent/`. Mechanical repo-wide reference update: every
   `-e ../X` relative install path stays correct automatically (sibling
   references, unaffected by the renamed directory's own name), but every
   doc, hardcoded path (e.g. `strategyqa_agent/ReAct/local_wikienv.py`'s
   hardcoded `"ReAct/database/..."` literals, which are relative to CWD at
   runtime and unaffected by the parent rename, but should be
   double-checked), and cross-reference needs updating.

4. **Phase 7d — documentation pass.** `CLAUDE.md`, root `README.md`, and
   each subproject's `README.md` rewritten to describe the final
   structure coherently in one pass, rather than patched incrementally
   the way the last six phases' final reviews caught doc gaps one at a
   time.

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
