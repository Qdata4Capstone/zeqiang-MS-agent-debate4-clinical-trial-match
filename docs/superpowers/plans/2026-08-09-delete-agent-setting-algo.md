# Delete Agent_Setting/algo/ (Phase 6) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete `Agent_Setting/algo/` (`trigger_optimization.py`, `config.py`, `utils.py`) — the gradient-guided adversarial trigger-optimization code the original brainstorming session ruled out of scope ("not RAG defense, not a simple attack") — and update the three documentation surfaces that describe it, so the repo's docs match its actual contents.

**Architecture:** Unlike Phases 1-5, this phase is pure deletion plus documentation, not extraction — there is no shared module to build and no adapter/parity-test pattern to follow. `algo/` is confirmed to have zero importers outside itself (verified by the controller before writing this plan via `grep -rln "from algo\.\|import algo\b" Agent_Setting --include="*.py"`, empty result). This plan does NOT touch `Agent_Setting/environment.yml`'s dependency list — several packages in that file (`wandb`, `autogen`, `pyautogen`, `gym`, `casadi`, `shapely`, `wolframalpha`, etc.) may have been needed only by `algo/trigger_optimization.py`, but determining that requires careful per-package usage analysis across the remaining `ReAct/` code, which belongs in Phase 7's `vulture`-based dead-code sweep (the next and final phase in the design spec), not this deletion.

**Tech Stack:** N/A (deletion + Markdown documentation only).

## Global Constraints

- Only delete `Agent_Setting/algo/trigger_optimization.py`, `Agent_Setting/algo/config.py`, and `Agent_Setting/algo/utils.py`. Do not touch anything under `Agent_Setting/ReAct/`.
- Do not touch `Agent_Setting/environment.yml` — dependency pruning is out of scope for this task, deferred to Phase 7.
- Before deleting, re-confirm via grep that nothing outside `algo/` imports from it — this is the one safety check that must not be skipped for a deletion task.
- Update `CLAUDE.md` and `Agent_Setting/README.md` so neither references `algo/trigger_optimization.py`/`algo/config.py`/`algo/utils.py` as if they still exist.
- Run all commands from the `safematch_v3` worktree root: `/Users/qiyanjun/Code/Public/zeqiang-MS-agent-debate4-clinical-trial-match/.worktrees/safematch_v3`.

---

### Task 1: Delete `Agent_Setting/algo/` and update its documentation

**Files:**
- Delete: `Agent_Setting/algo/trigger_optimization.py`
- Delete: `Agent_Setting/algo/config.py`
- Delete: `Agent_Setting/algo/utils.py`
- Modify: `CLAUDE.md`
- Modify: `Agent_Setting/README.md`

**Interfaces:** none — this task removes code and updates prose, it doesn't produce anything later tasks consume.

- [ ] **Step 1: Re-confirm nothing outside `algo/` imports from it**

Run:
```bash
grep -rln "from algo\.\|import algo\b\|algo\.config\|algo\.utils\|algo\.trigger_optimization" Agent_Setting --include="*.py" | grep -v "^Agent_Setting/algo/"
```
Expected: no output. If this produces any output, STOP — do not delete `algo/` — and report DONE_WITH_CONCERNS or NEEDS_CONTEXT describing what you found instead, since the plan's premise (nothing outside `algo/` depends on it) would be wrong.

- [ ] **Step 2: Delete the three files**

```bash
git rm Agent_Setting/algo/trigger_optimization.py Agent_Setting/algo/config.py Agent_Setting/algo/utils.py
```

(This also removes any `__pycache__` bytecode under `Agent_Setting/algo/` as an untracked side effect of the directory becoming empty — no separate step needed; if stray `__pycache__` files remain on disk after `git rm`, that's fine, they're already git-ignored per the repo's `.gitignore`.)

- [ ] **Step 3: Update `CLAUDE.md`**

Two edits in `CLAUDE.md`:

**Edit 3a** — the top-level "Repository overview" bullet for `Agent_Setting/` (currently line 11) reads:

```
- `Agent_Setting/` — adversarial trigger optimization against dense retrievers (AgentPoison-style) plus a ReAct agent (StrategyQA) with DRS defense and baseline comparisons.
```

Replace with:

```
- `Agent_Setting/` — a ReAct agent (StrategyQA) with DRS defense and baseline comparisons.
```

(The AgentPoison-style trigger-optimization description is removed since `algo/` — the only code implementing it — is deleted by this plan.)

**Edit 3b** — the `### Agent_Setting/` section (currently reading "Two independent pieces:" followed by an `algo/trigger_optimization.py` bullet and a `ReAct/run_strategyqa_inference.py` bullet):

```
Two independent pieces:
- `algo/trigger_optimization.py` — adversarial trigger optimization against a dense retriever (gradient-guided, perplexity-filtered), e.g. targeting `dpr-ctx_encoder-single-nq-base`. `algo/config.py` maps model codes to embedder HF repo names/paths (most are commented out; only `dpr-ctx_encoder-single-nq-base` is currently active). `algo/utils.py` holds shared helpers.
- `ReAct/run_strategyqa_inference.py` — runs a ReAct agent over StrategyQA (data in `ReAct/database/`, prompts in `ReAct/prompts/prompts.json`) with `--backbone qwen` (Ollama) and `--model dpr`, optionally with `--enable_drs` and `--compare_defenses` against baseline defenses in `defense_baselines.py`. `drs.py` implements the DRS defense used here; `local_wikienv.py`/`wrappers.py`/`search.py` implement the ReAct environment; `ollama_client.py` wraps Ollama calls.
```

Replace with (the "Two independent pieces:" framing no longer applies — there's only one piece left):

```
- `ReAct/run_strategyqa_inference.py` — runs a ReAct agent over StrategyQA (data in `ReAct/database/`, prompts in `ReAct/prompts/prompts.json`) with `--backbone qwen` (Ollama) and `--model dpr`, optionally with `--enable_drs` and `--compare_defenses` against baseline defenses in `defense_baselines.py`. `drs.py` implements the DRS defense used here; `local_wikienv.py`/`wrappers.py`/`search.py` implement the ReAct environment; `ollama_client.py` wraps Ollama calls.
```

Read the file first to confirm these are still the exact current lines before editing (this plan was written before Task 1 runs, so line numbers/content should match, but verify).

- [ ] **Step 4: Update `Agent_Setting/README.md`**

Read the file first. It currently has a `## 🚀 Quick Start` section with two numbered subsections: `### 1. Trigger Optimization` (a `python algo/trigger_optimization.py ...` example) and `### 2. ReAct-StrategyQA with DRS and Defense Baselines`. Remove the entire `### 1. Trigger Optimization` subsection (its heading, description line, and code block), and renumber `### 2. ReAct-StrategyQA with DRS and Defense Baselines` to `### 1. ReAct-StrategyQA with DRS and Defense Baselines`. Leave the `## 🛠️ Setup` section and the `## Notes` section at the end untouched.

- [ ] **Step 5: Run the full Agent_Setting test suite**

Run: `pytest Agent_Setting/tests/ -v`
Expected: all tests PASS (this suite doesn't test `algo/` at all — its three test files are all DRS/ollama-client parity tests for `ReAct/` — so this run just confirms the deletion didn't accidentally break an unrelated import path)

- [ ] **Step 6: Verify scope**

Run: `git status --porcelain`
Expected: shows exactly the three deletions (`Agent_Setting/algo/trigger_optimization.py`, `Agent_Setting/algo/config.py`, `Agent_Setting/algo/utils.py`) and two modifications (`CLAUDE.md`, `Agent_Setting/README.md`) — nothing else. (Pre-existing unrelated stale `.pyc` working-tree noise, if present, is not part of this task's scope — leave it unstaged.)

- [ ] **Step 7: Commit**

```bash
git add -A Agent_Setting/algo CLAUDE.md Agent_Setting/README.md
git commit -m "chore(Agent_Setting): delete algo/ trigger-optimization code, update docs

Removes trigger_optimization.py, config.py, and utils.py -- gradient-
guided adversarial trigger optimization against a dense retriever,
ruled out of scope during the original refactor brainstorming (not
RAG defense, not a simple attack). Confirmed nothing outside algo/
imported from it before deleting."
```

---

### Task 2: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none.

- [ ] **Step 1: Run every test suite in the repo**

```bash
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest RAG_Setting/tests/ -v
pytest Agent_Setting/tests/ -v
pytest Retrieving_stage/tests/ -v
```

Expected: all PASS — confirms deleting `algo/` didn't regress anything in any subproject.

- [ ] **Step 2: Confirm `algo/` is gone and nothing references it**

```bash
ls Agent_Setting/algo 2>&1
grep -rn "algo/trigger_optimization\|algo/config\|algo/utils\|algo\.trigger_optimization\|algo\.config\|algo\.utils" --include="*.py" --include="*.md" .  2>/dev/null | grep -v "docs/superpowers/specs\|docs/superpowers/plans"
```

Expected: `ls` reports "No such file or directory" (or, if git left an empty tracked-nothing directory stub, an empty listing); the grep returns no output outside the plan/spec docs themselves (which legitimately discuss the deletion historically).

- [ ] **Step 3: Report results to the user**

Summarize: what was deleted, which docs were updated, all test results, and confirm `Agent_Setting/environment.yml`'s dependency list was deliberately left untouched (deferred to Phase 7's dead-code sweep, which is better positioned to determine which packages are now genuinely unused). No commit needed for this task (verification only).
