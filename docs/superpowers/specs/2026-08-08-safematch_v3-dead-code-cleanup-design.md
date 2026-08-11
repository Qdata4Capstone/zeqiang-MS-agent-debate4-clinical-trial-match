# safematch_v3 dead-code cleanup — design

## Context

`safematch_v3` is a new branch/worktree, forked from `drs-shared-module`'s tip
(`0052ac1`), which already fully contains `safematch_v2` (other defense
baselines: `l2_norm`, `l2_distance`, `perplexity`, `defense_baselines.py`,
ReAct/StrategyQA setup, `poisonrag_experiment/`) plus the newer `drs_defense`
shared-module refactor.

The user asked for a "thorough analysis" of `safematch_v3` followed by removal
of "all functions not related to RAG defense." An initial repo-wide catalog
(via an Explore pass) showed that a literal reading of that request would
delete infrastructure the defenses depend on: attack-generation code
(`poisonedrag_blackbox.py`, `algo/trigger_optimization.py`) is what the
defenses are evaluated against, and retrieval/agent/LLM-client/data-loading
code is what every experiment (including defense ones) runs on top of.

Scope was clarified with the user: **cleanup only**. Remove genuine dead/stray
code (unused functions, orphaned files), not entire categories of
functionality. Attack code, retrieval code, agent orchestration code, and all
defense code stay.

## Goal

Do a systematic, tool-based dead-code scan of the repo's tracked Python
source and remove code that is genuinely unreferenced, without touching
anything that is reachable (directly, via CLI entry point, via dynamic
dispatch, or via test suite).

## Method

1. **Scan.** Run `vulture` across the four tracked Python trees:
   `drs_defense/`, `Retrieving_stage/`, `RAG_Setting/`, `Agent_Setting/`.
   Exclude `tests/`, `__pycache__/`, `*.egg-info/`, and dataset/JSON
   directories. Use a reasonably low confidence threshold first pass, then
   manually triage every hit — `vulture` alone is not a removal authority.

2. **Triage.** For each flagged item, check before removing:
   - Is it invoked only via CLI (`argparse`, `if __name__ == "__main__"`)?
   - Is it re-exported from an `__init__.py` for external/adapter use (e.g.
     `drs_defense` being consumed by the three thin adapters)?
   - Is it a pytest fixture, dunder method, or ABC-required override
     (`BaseDetector` subclasses in `RAG_Setting/.../defense/`)?
   - Is it referenced dynamically (string-based dispatch, config-driven
     method lookup)?
   - Is it part of the public surface of a module meant to be imported by
     the *other* two subprojects' adapters (per `drs_defense`'s stated
     shared-module role)?

   Anything matching the above is a false positive and is kept. Only
   confirmed-dead code is removed.

3. **Known removal.** `RAG_Setting/src/medrag_repro/defense/perpel` — a
   0-byte stray file already identified during the initial catalog, unrelated
   to any module. Removed regardless of `vulture` output.

4. **Out of scope.** `__pycache__/`, `.DS_Store`, `*.egg-info/` — already
   git-ignored and untracked (per the prior "untrack generated artifacts"
   commit on this line of history). These are not a code change; any local
   tidy-up of them is housekeeping only, not part of the commit.

5. **Verify.** After removals, run:
   - `drs_defense/tests/` (the paper-parity suite)
   - `Agent_Setting/tests/`, `RAG_Setting/tests/`, `Retrieving_stage/tests/`
     (the three DRS-adapter parity suites)

   All must still pass. If a removal breaks a test, that item was not
   actually dead — restore it and re-triage.

6. **Commit.** One commit on `safematch_v3` containing the removed dead code,
   with a commit message listing what was removed and, briefly, why each item
   was judged dead (since `vulture` hits alone aren't self-explanatory in a
   commit log).

## Non-goals

- No removal of attack-generation code (`poisonedrag_blackbox.py`,
  `algo/trigger_optimization.py`, `algo/utils.py`, `algo/config.py`).
- No removal of core retrieval, agent-environment, LLM-client, or
  data-loading code.
- No refactors, renames, or reorganization beyond deletion of confirmed-dead
  code.
- No changes to `drs_defense/` core math or its adapters' behavior.

## Risks

- `vulture` false positives on dynamically-dispatched or externally-imported
  code are the main risk of an over-aggressive removal — mitigated by the
  manual triage step and by running all four test suites before committing.
- Some "dead" code may be intentionally kept for a not-yet-wired experiment
  (e.g., ablation flags). Where intent is unclear from the code alone, it
  will be flagged to the user rather than removed silently.

## Outcome (implemented)

Executed on 2026-08-10, against the current post-Phase-7a–7c structure
(`drs_defense/`, `infra/`, `attacks/`, `defenses/`, `use-cases/*` — this
spec's own paths above are stale, written before that rename). Method
followed exactly: `vulture` (confidence 60, then 40 for a broader recheck)
across every shared library and use case's code (tests/ excluded), every
hit manually triaged against the checklist above before removal.

Removed: the `perpel` known-removal (finally executed — flagged here since
2026-08-08 but never actually done until this pass); a whole unused
`Prompter` class (`use-cases/strategyqa_agent/ReAct/utils/prompter.py`);
three unused, two of them provably broken (`MMLU_SPLIT_FILE`/
`HOTPOTQA_SPLIT_FILE` undefined anywhere), leftover multi-task wrapper
classes in `wrappers.py` (`HistoryWrapper`, `MMLUWrapper`,
`HotPotQAWrapper`) plus their cascading-dead-only-callers
(`f1_score`, `normalize_answer`, and the `re`/`string`/`pandas`/`Counter`
imports that existed only to support them); an unused `ast` import and
`get_time_info` method in `local_wikienv.py`, plus its cascading-dead
`search_time`/`num_searches` instrumentation and `import time`.

One real false positive caught and instructive: `infra/`'s
`ollama_completion` flagged as unused when `infra/src` was scanned in
isolation, but confirmed actively used (re-exported by
`strategyqa_agent/ReAct/ollama_client.py`, tested twice, called directly)
once `strategyqa_agent/ReAct` was included in the same `vulture` pass —
exactly the "manual triage catches narrow-scope false positives" risk this
spec anticipated, observed concretely rather than just theorized.

Ambiguous-intent local variables in two otherwise-live scripts
(`ReAct/eval.py`, `ReAct/run_strategyqa_inference.py`) were deliberately
left alone per the "flag rather than remove silently" risk note above,
not because they're false positives but because removing them adds little
value against real (if small) risk of misreading intent in commented-out,
actively-iterated-on code.

All 86 tests (4 shared libraries + 3 use cases) still pass; a second
`vulture` pass after the edits found no further cascading dead code.
See commit `008f736`.
