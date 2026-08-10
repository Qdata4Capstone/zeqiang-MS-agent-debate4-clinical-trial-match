# Rename Subprojects (Phase 7c) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the three top-level subproject directories to reflect what they actually run, per explicit user decision: `Retrieving_stage/` → `trial_retrieval/`, `RAG_Setting/` → `medqa_rag/`, `Agent_Setting/` → `strategyqa_agent/`. **Directories only** — the Python packages inside them (`medrag_repro`, `poisonrag_experiment`, `ReAct`) keep their current import names; nothing imports a directory name, so this touches zero import statements. Scope confirmed narrow by explicit user choice over the alternative (also renaming the Python packages, a much larger effort).

**Architecture:** This is the most mechanical phase so far — no new shared code, no adapters, no parity tests. Research before writing this plan (full repo-wide grep, both `.py` and doc/config files, excluding `docs/superpowers/` which is a historical record and stays as-is) found exactly two categories of change needed:
1. **One functionally-required fix**: `attacks/tests/conftest.py` hardcodes the literal string `"Retrieving_stage"` in a `sys.path.insert` call (added during Phase 7a's final-review fix) — this MUST change to `"trial_retrieval"` or `attacks/tests/test_poisonedrag_trial.py` will fail to import `poisonrag_experiment` after the rename.
2. **Nine files with prose-only references** (no functional effect, but must stay accurate): root `README.md`, `CLAUDE.md`, `defenses/README.md`, `defenses/pyproject.toml`, `infra/pyproject.toml`, `infra/README.md`, `drs_defense/README.md`, `attacks/pyproject.toml`, `attacks/README.md`.

One more functionally-required step, found during research (not obvious from grep, requires understanding of how Python editable installs work): `RAG_Setting/requirements.txt` has an `-e .` line that installs `RAG_Setting` itself as the `medrag-repro` package. `pip install -e .` bakes the **absolute path** of the source directory into the installed package's metadata (`Editable project location: .../RAG_Setting`, confirmed via `pip show -f medrag-repro` before writing this plan). Renaming the directory breaks this editable install — `import medrag_repro` will fail with the package's finder pointing at a now-nonexistent path — until the package is reinstalled from its new location (`pip install -e medqa_rag`). `Retrieving_stage`/`Agent_Setting` have no equivalent self-install (`poisonrag_experiment`/`ReAct` are imported via pytest-rootdir `sys.path` convention, not pip), so only `medrag-repro` needs reinstalling.

**Tech Stack:** N/A (directory rename, text edits, one `pip install -e` reinstall).

**Mid-execution addition (after Task 1 completed):** the user asked to also nest the three renamed directories under a new top-level `use-cases/` directory, separating them from the four shared-library packages (`drs_defense/`, `infra/`, `attacks/`, `defenses/`), which stay at the repo root. This became Task 2, inserted before the original documentation task (now Task 3) so the doc pass writes the final paths once rather than twice. Task 2's own section explains its specific impact (ten relative install-path lines, one more `conftest.py` fix, one more `medrag-repro` reinstall).

## Global Constraints

- Directories only — do NOT rename `medrag_repro`, `poisonrag_experiment`, or `ReAct` (the Python packages inside the renamed/nested directories), and do NOT change any `pyproject.toml`'s `name =`/`[project]` field for `medrag-repro` itself.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` are historical records — leave every reference to `RAG_Setting`/`Agent_Setting`/`Retrieving_stage` in those files exactly as it is. Confirmed via research: no file outside `docs/superpowers/` other than the nine listed in Task 3 references the old names in prose, and no `.py` file outside `attacks/tests/conftest.py` hardcodes an old directory name as a string.
- After Task 1's rename, `medrag-repro`'s editable install MUST be refreshed (`pip install -e medqa_rag`); after Task 2's nesting, it must be refreshed AGAIN from the new location (`pip install -e use-cases/medqa_rag`) — this is not optional, `import medrag_repro` will fail otherwise each time. `Retrieving_stage`/`trial_retrieval` and `Agent_Setting`/`strategyqa_agent` have no equivalent self-install and need no reinstall step at either stage.
- Task 1's rename alone does not require touching any `-e ../X` line (sibling-relative paths, unaffected by a renamed-but-still-sibling directory's own name). Task 2's nesting DOES require updating every `-e ../X` line to `-e ../../X` (ten lines across three files) — see Task 2 for the exact list.
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

Expected: all PASS (same counts as before the rename — 19, 14, 17, 14, 5, 7, 6 respectively = 82 total — confirming nothing broke; `infra/tests/` is 17, not 19, since commit `ccb9e7e` legitimately dropped an orphaned test earlier in Phase 7b, before this plan was written).

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

**Note on execution:** Task 1 actually landed across two commits (`e48ff88`, `392fd18`) rather than one, due to a controller-side git mistake during execution (a docs-fix commit accidentally swept up the already-staged `git mv` renames). The end state — three directories renamed, `conftest.py` fixed, `medrag-repro` reinstalled, all 82 tests passing — matches this task's intent exactly; only the commit history shape differs from what this section originally specified. See the SDD ledger for the full account. This note is left here for anyone reading this plan after the fact.

---

### Task 2: Nest the three renamed directories under `use-cases/`

**Added mid-execution, per user request after Task 1 completed**: group the three use-case/experiment directories (`medqa_rag/`, `strategyqa_agent/`, `trial_retrieval/`) under a new top-level `use-cases/` directory, separating them from the four shared-library packages (`drs_defense/`, `infra/`, `attacks/`, `defenses/`), which stay at the repo root. This changes the target layout the original Task 2 (now Task 3) documents against, so it must run first.

**Files:**
- Rename (via `git mv`, into a newly created `use-cases/` directory): `medqa_rag/` → `use-cases/medqa_rag/`, `strategyqa_agent/` → `use-cases/strategyqa_agent/`, `trial_retrieval/` → `use-cases/trial_retrieval/`
- Modify: `use-cases/medqa_rag/requirements.txt` (4 lines: `-e ../drs_defense`, `-e ../infra`, `-e ../attacks`, `-e ../defenses` → `-e ../../X`)
- Modify: `use-cases/trial_retrieval/requirements.txt` (3 lines: `-e ../drs_defense`, `-e ../infra`, `-e ../attacks` → `-e ../../X`)
- Modify: `use-cases/strategyqa_agent/environment.yml` (3 lines: `- -e ../drs_defense`, `- -e ../infra`, `- -e ../defenses` → `- -e ../../X`)
- Modify: `attacks/tests/conftest.py`

**Interfaces:** none — relocates existing code, doesn't produce new interfaces.

- [ ] **Step 1: Create the `use-cases/` directory and move the three subprojects into it**

```bash
mkdir -p use-cases
git mv medqa_rag use-cases/medqa_rag
git mv strategyqa_agent use-cases/strategyqa_agent
git mv trial_retrieval use-cases/trial_retrieval
```

- [ ] **Step 2: Fix the ten now-broken relative `-e ../X` install paths**

Each of the three moved subprojects is now one directory level deeper, so every `-e ../X` line pointing at a shared package needs an extra `../`. Read each file first to confirm current content matches what's described below before editing.

In `use-cases/medqa_rag/requirements.txt`, change:
```
-e .
-e ../drs_defense
-e ../infra
-e ../attacks
-e ../defenses
pytest
```
to:
```
-e .
-e ../../drs_defense
-e ../../infra
-e ../../attacks
-e ../../defenses
pytest
```
(Note: `-e .` stays unchanged — it installs the package from its own current directory, unaffected by nesting depth.)

In `use-cases/trial_retrieval/requirements.txt`, change the three `-e ../X` lines (`../drs_defense`, `../infra`, `../attacks`) to `-e ../../X`, leaving every pinned package line above them and the `pytest` line below them untouched.

In `use-cases/strategyqa_agent/environment.yml`, change the three `- -e ../X` lines (`../drs_defense`, `../infra`, `../defenses`) to `- -e ../../X`, leaving the rest of the `pip:` block untouched.

- [ ] **Step 3: Fix `attacks/tests/conftest.py`'s path depth**

The file currently does:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "trial_retrieval"))
```
`trial_retrieval` is now nested one level deeper (`use-cases/trial_retrieval`), so change this to:
```python
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "use-cases" / "trial_retrieval"))
```
Update the file's docstring too, if it references the bare `trial_retrieval/` path, to mention `use-cases/trial_retrieval/`.

- [ ] **Step 4: Reinstall `medrag-repro` from its new (doubly-new) location**

Run: `pip install -e use-cases/medqa_rag`
Verify: `python3 -c "import medrag_repro; print(medrag_repro.__file__)"` — path should now start with `.../safematch_v3/use-cases/medqa_rag/...`

- [ ] **Step 5: Run every test suite at its new path**

```bash
pytest defenses/tests/ -v
pytest attacks/tests/ -v
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest use-cases/medqa_rag/tests/ -v
pytest use-cases/strategyqa_agent/tests/ -v
pytest use-cases/trial_retrieval/tests/ -v
```
Expected: all PASS, same 82-test total (19/14/17/14/5/7/6) as before this task.

- [ ] **Step 6: Verify scope and commit**

```bash
git status --porcelain
```
Expected: the three nested-directory renames, the three modified dependency-manifest files, and `attacks/tests/conftest.py` — nothing else.

```bash
git add -A use-cases attacks/tests/conftest.py
git commit -m "chore: nest medqa_rag/strategyqa_agent/trial_retrieval under use-cases/

Separates the three use-case/experiment directories from the four
shared-library packages (drs_defense/, infra/, attacks/, defenses/),
which stay at the repo root. Fixed the ten relative -e ../X install
paths (now -e ../../X, one level deeper) and attacks/tests/conftest.py's
hardcoded path depth. Reinstalled medrag-repro from its new location."
```

---

### Task 3: Update documentation and config prose references

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

**Note — target strings changed after Task 2 was added mid-execution.** These nine files never got updated during Task 1 or Task 2 (both touched only directories/manifests/code, not prose) — they still contain the ORIGINAL pre-rename names (`RAG_Setting`, `Agent_Setting`, `Retrieving_stage`). This task substitutes directly from those original names to their final, doubly-updated location under `use-cases/` — there is no intermediate `medqa_rag/`-without-`use-cases/` state to document, since Task 2 already nested them before this task runs.

- [ ] **Step 1: Replace old directory names with their final `use-cases/`-nested paths in all nine files**

In each of the nine files listed above, replace every whole-word occurrence of:
- `RAG_Setting` → `use-cases/medqa_rag`
- `Agent_Setting` → `use-cases/strategyqa_agent`
- `Retrieving_stage` → `use-cases/trial_retrieval`

This can be done with `sed` for speed, but READ each file's diff afterward — don't trust a blind substitution without checking the result, since Markdown link syntax needs both the link text and the href updated consistently (e.g. `` [`Retrieving_stage/`](Retrieving_stage/README.md) `` must become `` [`use-cases/trial_retrieval/`](use-cases/trial_retrieval/README.md) ``, not just one half of it):

```bash
for f in README.md CLAUDE.md defenses/README.md defenses/pyproject.toml infra/pyproject.toml infra/README.md drs_defense/README.md attacks/pyproject.toml attacks/README.md; do
  sed -i '' \
    -e 's/RAG_Setting/use-cases\/medqa_rag/g' \
    -e 's/Agent_Setting/use-cases\/strategyqa_agent/g' \
    -e 's/Retrieving_stage/use-cases\/trial_retrieval/g' \
    "$f"
done
```

(The `sed -i ''` empty-string argument is macOS/BSD sed's in-place-edit syntax — this repo's dev environment is macOS per the session context. If running on GNU sed, use `sed -i` without the trailing `''`. The `/` in the replacement text is escaped as `\/` since `sed`'s default delimiter is also `/`.)

Do NOT run this substitution against any other file — in particular, do NOT touch `docs/superpowers/specs/` or `docs/superpowers/plans/` (historical records, must keep the old names as they accurately describe what existed at each point in time), and do NOT touch any `.py` file with this script (Tasks 1-2 already handled the two `.py` files that needed changes).

- [ ] **Step 2: Read every changed file's diff and confirm it reads correctly**

Run: `git diff README.md CLAUDE.md defenses/README.md defenses/pyproject.toml infra/pyproject.toml infra/README.md drs_defense/README.md attacks/pyproject.toml attacks/README.md`

Check specifically:
- Every Markdown link (`[`text`](path)`) has BOTH its visible text and its href path updated consistently, and the href actually resolves (e.g. `use-cases/trial_retrieval/README.md` must be a real file after Task 2's move — verify with `ls`).
- No sentence reads awkwardly or ends up self-contradictory after the substitution (read each changed line in context, not just the diff hunk) — e.g. "Setup (from `use-cases/medqa_rag/`):" should still read naturally.
- `medrag_repro`, `poisonrag_experiment`, `ReAct` (the Python package names, NOT directory names) are unaffected — the substitution patterns above don't match these strings, so this should already be true, but verify.

- [ ] **Step 3: Verify no stale references remain outside the historical docs**

```bash
grep -rl "RAG_Setting\|Agent_Setting\|Retrieving_stage" --include="*.md" --include="*.yml" --include="*.toml" --include="*.txt" --include="*.json" . 2>/dev/null | grep -v "docs/superpowers"
```

Expected: no output.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md defenses/README.md defenses/pyproject.toml infra/pyproject.toml infra/README.md drs_defense/README.md attacks/pyproject.toml attacks/README.md
git commit -m "docs: update subproject directory references after the Phase 7c rename + use-cases/ nesting"
```

---

### Task 4: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none.

- [ ] **Step 1: Run every test suite one more time**

```bash
pytest defenses/tests/ -v
pytest attacks/tests/ -v
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest use-cases/medqa_rag/tests/ -v
pytest use-cases/strategyqa_agent/tests/ -v
pytest use-cases/trial_retrieval/tests/ -v
```

Expected: all PASS (same 82-test total as before this phase — 19/14/17/14/5/7/6).

- [ ] **Step 2: Confirm the old directory names are gone from the filesystem and from git tracking, and the new nested layout is correct**

```bash
ls RAG_Setting Agent_Setting Retrieving_stage medqa_rag strategyqa_agent trial_retrieval 2>&1
git ls-files | grep -E "^(RAG_Setting|Agent_Setting|Retrieving_stage)/" | head -5
ls use-cases/
```

Expected: the first `ls` reports "No such file or directory" for all six paths (the three original names AND the three un-nested intermediate names — none should exist, since Task 2 moved them into `use-cases/`); `git ls-files` returns no output; `ls use-cases/` shows exactly `medqa_rag`, `strategyqa_agent`, `trial_retrieval`.

- [ ] **Step 3: Confirm zero stale references anywhere outside `docs/superpowers/`**

```bash
grep -rl "RAG_Setting\|Agent_Setting\|Retrieving_stage" . 2>/dev/null | grep -v "docs/superpowers" | grep -v __pycache__ | grep -v egg-info | grep -v "\.git/"
```

Expected: no output.

Also confirm no doc surface references the three subprojects at the repo root without the `use-cases/` prefix (a leftover from Task 3's substitution not accounting for the nesting would show up here):

```bash
grep -rn "\`medqa_rag/\|\`strategyqa_agent/\|\`trial_retrieval/" README.md CLAUDE.md 2>/dev/null | grep -v "use-cases/"
```

Expected: no output (every mention should be prefixed `use-cases/`).

- [ ] **Step 4: Verify `medrag_repro` imports correctly from its new location**

Run: `python3 -c "import medrag_repro; print(medrag_repro.__file__)"`
Expected: path starts with `.../safematch_v3/use-cases/medqa_rag/...`

- [ ] **Step 5: Report results to the user**

Summarize: the three renames, the `use-cases/` nesting (added mid-execution per user request), the functional fixes (`attacks/tests/conftest.py` fixed twice — once for the rename, once for the nesting depth — and `medrag-repro` reinstalled twice for the same reason), the nine documentation files updated to the final nested paths, and confirm all 82 tests still pass with zero stale references outside the historical `docs/superpowers/` record. No commit needed for this task (verification only).
