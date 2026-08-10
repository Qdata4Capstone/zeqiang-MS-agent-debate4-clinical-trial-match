# Rename Subprojects (Phase 7c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the three top-level subproject directories to reflect what they actually run, per explicit user decision: `Retrieving_stage/` → `trial_retrieval/`, `RAG_Setting/` → `medqa_rag/`, `Agent_Setting/` → `strategyqa_agent/`. **Directories only** — the Python packages inside them (`medrag_repro`, `poisonrag_experiment`, `ReAct`) keep their current import names; nothing imports a directory name, so this touches zero import statements. Scope confirmed narrow by explicit user choice over the alternative (also renaming the Python packages, a much larger effort).

**Architecture:** This is the most mechanical phase so far — no new shared code, no adapters, no parity tests. Research before writing this plan (full repo-wide grep, both `.py` and doc/config files, excluding `docs/superpowers/` which is a historical record and stays as-is) found exactly two categories of change needed:
1. **One functionally-required fix**: `attacks/tests/conftest.py` hardcodes the literal string `"Retrieving_stage"` in a `sys.path.insert` call (added during Phase 7a's final-review fix) — this MUST change to `"trial_retrieval"` or `attacks/tests/test_poisonedrag_trial.py` will fail to import `poisonrag_experiment` after the rename.
2. **Nine files with prose-only references** (no functional effect, but must stay accurate): root `README.md`, `CLAUDE.md`, `defenses/README.md`, `defenses/pyproject.toml`, `infra/pyproject.toml`, `infra/README.md`, `drs_defense/README.md`, `attacks/pyproject.toml`, `attacks/README.md`.

One more functionally-required step, found during research (not obvious from grep, requires understanding of how Python editable installs work): `RAG_Setting/requirements.txt` has an `-e .` line that installs `RAG_Setting` itself as the `medrag-repro` package. `pip install -e .` bakes the **absolute path** of the source directory into the installed package's metadata (`Editable project location: .../RAG_Setting`, confirmed via `pip show -f medrag-repro` before writing this plan). Renaming the directory breaks this editable install — `import medrag_repro` will fail with the package's finder pointing at a now-nonexistent path — until the package is reinstalled from its new location (`pip install -e medqa_rag`). `Retrieving_stage`/`Agent_Setting` have no equivalent self-install (`poisonrag_experiment`/`ReAct` are imported via pytest-rootdir `sys.path` convention, not pip), so only `medrag-repro` needs reinstalling.

**Tech Stack:** N/A (directory rename, text edits, one `pip install -e` reinstall).

## Global Constraints

- Directories only — do NOT rename `medrag_repro`, `poisonrag_experiment`, or `ReAct` (the Python packages inside the renamed directories), and do NOT change any `pyproject.toml`'s `name =`/`[project]` field for `medrag-repro` itself.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` are historical records — leave every reference to `RAG_Setting`/`Agent_Setting`/`Retrieving_stage` in those files exactly as it is. Confirmed via research: no file outside `docs/superpowers/` other than the nine listed above references the old names in prose, and no `.py` file outside `attacks/tests/conftest.py` hardcodes an old directory name as a string.
- After renaming, `medrag-repro`'s editable install MUST be refreshed (`pip install -e medqa_rag`) — this is not optional, `import medrag_repro` will fail otherwise. `Retrieving_stage` and `Agent_Setting` have no equivalent self-install and need no reinstall step.
- Every `-e ../X` relative install path (in each subproject's `requirements.txt`/`environment.yml`, pointing at `drs_defense`/`infra`/`attacks`/`defenses`) resolves correctly automatically after the rename — these are sibling-relative paths, unaffected by the renamed directory's own name. No changes needed to any `-e ../X` line anywhere.
- Run all commands from the `safematch_v3` worktree root: `/Users/qiyanjun/Code/Public/zeqiang-MS-agent-debate4-clinical-trial-match/.worktrees/safematch_v3`.

---

### Task 1: Rename directories, fix the functional breakage, reinstall, verify

**Files:**
- Rename (via `git mv`): `Retrieving_stage/` → `trial_retrieval/`
- Rename (via `git mv`): `RAG_Setting/` → `medqa_rag/`
- Rename (via `git mv`): `Agent_Setting/` → `strategyqa_agent/`
- Modify: `attacks/tests/conftest.py`

**Interfaces:** none — this task only relocates existing code, it doesn't produce new interfaces.

- [ ] **Step 1: Rename the three directories**

```bash
git mv Retrieving_stage trial_retrieval
git mv RAG_Setting medqa_rag
git mv Agent_Setting strategyqa_agent
```

- [ ] **Step 2: Fix `attacks/tests/conftest.py`'s hardcoded path**

Read the file first to confirm its current content matches this description. It currently contains (among a docstring explaining why the fixup exists):

```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Retrieving_stage"))
```

Change `"Retrieving_stage"` to `"trial_retrieval"`. Also update the docstring at the top of the file, which references `Retrieving_stage` by name in its explanation — reword those references to `trial_retrieval` so the comment matches the code.

- [ ] **Step 3: Reinstall `medrag-repro` from its new location**

Run: `pip install -e medqa_rag`
Expected: `Successfully installed medrag-repro-0.1.0` (reinstalled, not a fresh install — version stays 0.1.0)

Verify: `python3 -c "import medrag_repro; print(medrag_repro.__file__)"` — the printed path should now start with `.../safematch_v3/medqa_rag/...`, not `.../RAG_Setting/...`.

- [ ] **Step 4: Run every test suite at its new path**

```bash
pytest defenses/tests/ -v
pytest attacks/tests/ -v
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest medqa_rag/tests/ -v
pytest strategyqa_agent/tests/ -v
pytest trial_retrieval/tests/ -v
```

Expected: all PASS (same counts as before the rename — 19, 14, 19, 14, 5, 7, 6 respectively — confirming nothing broke).

- [ ] **Step 5: Verify scope**

Run: `git status --porcelain`
Expected: shows the three directory renames (git typically reports these as paired delete/add or as rename-detected `R` entries depending on similarity threshold) and the one modified file (`attacks/tests/conftest.py`) — nothing else.

- [ ] **Step 6: Commit**

```bash
git add -A trial_retrieval medqa_rag strategyqa_agent attacks/tests/conftest.py
git commit -m "chore: rename subproject directories (Retrieving_stage->trial_retrieval, RAG_Setting->medqa_rag, Agent_Setting->strategyqa_agent)

Directories only -- medrag_repro, poisonrag_experiment, and ReAct keep
their current Python import names. Reinstalled medrag-repro from its
new location (pip editable installs bake in an absolute path). Fixed
attacks/tests/conftest.py's hardcoded Retrieving_stage path reference,
the one functionally-required text change."
```

---

### Task 2: Update documentation and config prose references

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `defenses/README.md`
- Modify: `defenses/pyproject.toml`
- Modify: `infra/pyproject.toml`
- Modify: `infra/README.md`
- Modify: `drs_defense/README.md`
- Modify: `attacks/pyproject.toml`
- Modify: `attacks/README.md`

**Interfaces:** none — documentation/config-description text only.

- [ ] **Step 1: Replace old directory names with new ones in all nine files**

In each of the nine files listed above, replace every whole-word occurrence of:
- `RAG_Setting` → `medqa_rag`
- `Agent_Setting` → `strategyqa_agent`
- `Retrieving_stage` → `trial_retrieval`

This can be done with `sed` for speed, but READ each file's diff afterward — don't trust a blind substitution without checking the result, since Markdown link syntax needs both the link text and the href updated consistently (e.g. `` [`Retrieving_stage/`](Retrieving_stage/README.md) `` must become `` [`trial_retrieval/`](trial_retrieval/README.md) ``, not just one half of it):

```bash
for f in README.md CLAUDE.md defenses/README.md defenses/pyproject.toml infra/pyproject.toml infra/README.md drs_defense/README.md attacks/pyproject.toml attacks/README.md; do
  sed -i '' \
    -e 's/RAG_Setting/medqa_rag/g' \
    -e 's/Agent_Setting/strategyqa_agent/g' \
    -e 's/Retrieving_stage/trial_retrieval/g' \
    "$f"
done
```

(The `sed -i ''` empty-string argument is macOS/BSD sed's in-place-edit syntax — this repo's dev environment is macOS per the session context. If running on GNU sed, use `sed -i` without the trailing `''`.)

Do NOT run this substitution against any other file — in particular, do NOT touch `docs/superpowers/specs/` or `docs/superpowers/plans/` (historical records, must keep the old names as they accurately describe what existed at each point in time), and do NOT touch any `.py` file with this script (Task 1 already handled the one `.py` file that needed a change).

- [ ] **Step 2: Read every changed file's diff and confirm it reads correctly**

Run: `git diff README.md CLAUDE.md defenses/README.md defenses/pyproject.toml infra/pyproject.toml infra/README.md drs_defense/README.md attacks/pyproject.toml attacks/README.md`

Check specifically:
- Every Markdown link (`[`text`](path)`) has BOTH its visible text and its href path updated consistently.
- No sentence reads awkwardly or ends up self-contradictory after the substitution (read each changed line in context, not just the diff hunk).
- `medrag_repro`, `poisonrag_experiment`, `ReAct` (the Python package names, NOT directory names) are unaffected — the substitution patterns above don't match these strings, so this should already be true, but verify.

- [ ] **Step 3: Verify no stale references remain outside the historical docs**

```bash
grep -rl "RAG_Setting\|Agent_Setting\|Retrieving_stage" --include="*.md" --include="*.yml" --include="*.toml" --include="*.txt" --include="*.json" . 2>/dev/null | grep -v "docs/superpowers"
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md defenses/README.md defenses/pyproject.toml infra/pyproject.toml infra/README.md drs_defense/README.md attacks/pyproject.toml attacks/README.md
git commit -m "docs: update subproject directory references after the Phase 7c rename"
```

---

### Task 3: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none.

- [ ] **Step 1: Run every test suite one more time**

```bash
pytest defenses/tests/ -v
pytest attacks/tests/ -v
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest medqa_rag/tests/ -v
pytest strategyqa_agent/tests/ -v
pytest trial_retrieval/tests/ -v
```

Expected: all PASS (same 84-test total as before this phase).

- [ ] **Step 2: Confirm the old directory names are gone from the filesystem and from git tracking**

```bash
ls RAG_Setting Agent_Setting Retrieving_stage 2>&1
git ls-files | grep -E "^(RAG_Setting|Agent_Setting|Retrieving_stage)/" | head -5
```

Expected: `ls` reports "No such file or directory" for all three; `git ls-files` returns no output (nothing still tracked under the old paths).

- [ ] **Step 3: Confirm zero stale references anywhere outside `docs/superpowers/`**

```bash
grep -rl "RAG_Setting\|Agent_Setting\|Retrieving_stage" . 2>/dev/null | grep -v "docs/superpowers" | grep -v __pycache__ | grep -v egg-info | grep -v "\.git/"
```

Expected: no output.

- [ ] **Step 4: Verify `medrag_repro` imports correctly from its new location**

Run: `python3 -c "import medrag_repro; print(medrag_repro.__file__)"`
Expected: path starts with `.../safematch_v3/medqa_rag/...`

- [ ] **Step 5: Report results to the user**

Summarize: the three renames, the two functional fixes (`attacks/tests/conftest.py`, `medrag-repro` reinstall), the nine documentation files updated, and confirm all 84 tests still pass with zero stale references outside the historical `docs/superpowers/` record. No commit needed for this task (verification only).
