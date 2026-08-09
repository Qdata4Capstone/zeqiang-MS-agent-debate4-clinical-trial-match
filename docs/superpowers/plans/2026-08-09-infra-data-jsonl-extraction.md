# infra/data/ Extraction (Phase 2, narrowed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the four generic, dependency-free file-I/O helpers currently living inside `Retrieving_stage/poisonrag_experiment/retrieval_utils.py` (`load_jsonl`, `dump_json`, `load_qrels`, `load_queries_and_keywords`) into the shared `rag_infra` package as `rag_infra.data.jsonl`, following the same shared-implementation + thin re-export + parity-test pattern used in Phase 1.

**Architecture:** This is a narrowed version of the design spec's Phase 2 (`infra/data/`). Research before writing this plan found that RAG_Setting's `medqa_loader.py`/`pubmed_loader.py` are tightly coupled to subproject-specific dataclasses (`medrag_repro.datamodels.{QAItem, CorpusDoc}`) and `medrag_repro.utils.text.normalize_ws`, and Agent_Setting/ReAct's StrategyQA loading is a method on the stateful `WikiEnv` class — neither is a clean, standalone relocation like Phase 1's LLM clients. Only `Retrieving_stage`'s four JSON/JSONL/TSV file-I/O functions are genuinely standalone (stdlib-only: `json`, `os`, `csv`) and worth centralizing now. Unlike Phase 1 (where the whole subproject file became a re-export), here only 4 of ~15 functions in `retrieval_utils.py` move out — the rest of that file (BM25/MedCPT indexing, reciprocal-rank fusion) stays in place and is unaffected.

**Tech Stack:** Python stdlib (`json`, `os`, `csv`), `pytest` with `tmp_path` for real file-I/O tests (no mocking needed — these functions have no network or external-service dependency).

## Global Constraints

- `rag_infra` must stay importable under both Python 3.9 (Agent_Setting's pinned conda env) and Python 3.10 (RAG_Setting's conda env) — no syntax newer than 3.9. (This phase doesn't touch Agent_Setting, but the shared package as a whole must stay compatible.)
- No changes to call-site behavior: the sole existing caller, `Retrieving_stage/poisonrag_experiment/run_poisonrag_experiment.py`, imports `dump_json, load_jsonl, load_qrels, load_queries_and_keywords` from `poisonrag_experiment.retrieval_utils` (alongside other names — `MedCPTQueryEncoder`, `build_bm25_index`, `build_medcpt_corpus_index`, `get_conditions`, `rank_query`, `recall_at_k` — that are NOT moving and must keep resolving from `retrieval_utils.py` directly). That import statement must not need to change.
- `retrieval_utils.py`'s own internal use of `dump_json` (inside `build_medcpt_corpus_index`, currently at line 129) must keep working after the extraction.
- No changes to RAG_Setting's or Agent_Setting's data-loading code in this phase — out of scope per the narrowed plan.
- No new dependency needed: all four functions are stdlib-only, so `infra/pyproject.toml` does not need a new dependency for this phase.
- `Retrieving_stage/requirements.txt` already has `-e ../infra` (added in Phase 1) — no manifest change needed in this phase.
- Follow the `drs_defense`/Phase-1 precedent exactly: shared implementation + thin re-export + identity-based parity test (`is`, not reimplementation or behavioral-equivalence).
- Run all commands from the `safematch_v3` worktree root: `/Users/qiyanjun/Code/Public/zeqiang-MS-agent-debate4-clinical-trial-match/.worktrees/safematch_v3`. Use whatever `python3`/`pip` the shell resolves to — `rag_infra` is already installed editable in this environment from Phase 1.

---

### Task 1: `rag_infra.data.jsonl` (shared JSON/JSONL/TSV file-I/O helpers)

**Files:**
- Create: `infra/src/rag_infra/data/__init__.py`
- Create: `infra/src/rag_infra/data/jsonl.py`
- Test: `infra/tests/test_data_jsonl.py`

**Interfaces:**
- Produces: `rag_infra.data.jsonl.load_jsonl(path: str) -> list[dict]`
- Produces: `rag_infra.data.jsonl.dump_json(path: str, payload) -> None`
- Produces: `rag_infra.data.jsonl.load_qrels(dataset_dir: str) -> dict[str, dict[str, int]]`
- Produces: `rag_infra.data.jsonl.load_queries_and_keywords(dataset_dir: str) -> tuple[dict, dict]`

- [ ] **Step 1: Create the `data` subpackage**

Create `infra/src/rag_infra/data/__init__.py` (empty file).

- [ ] **Step 2: Write the failing tests**

Create `infra/tests/test_data_jsonl.py`:

```python
from __future__ import annotations

import json

from rag_infra.data.jsonl import (
    dump_json,
    load_jsonl,
    load_qrels,
    load_queries_and_keywords,
)


def test_load_jsonl_reads_lines_as_json_objects(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"_id": "d1", "text": "a"}\n{"_id": "d2", "text": "b"}\n')

    result = load_jsonl(str(p))

    assert result == [{"_id": "d1", "text": "a"}, {"_id": "d2", "text": "b"}]


def test_dump_json_writes_indented_json_and_round_trips(tmp_path):
    p = tmp_path / "out.json"

    dump_json(str(p), {"a": 1, "b": [1, 2, 3]})

    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}
    assert p.read_text() == json.dumps({"a": 1, "b": [1, 2, 3]}, indent=2)


def test_dump_json_creates_missing_parent_directories(tmp_path):
    p = tmp_path / "nested" / "deeper" / "out.json"

    dump_json(str(p), {"ok": True})

    assert p.exists()
    assert json.loads(p.read_text()) == {"ok": True}


def test_load_qrels_parses_tsv_into_nested_dict(tmp_path):
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td1\t1\n"
    )

    result = load_qrels(str(tmp_path))

    assert result == {"q1": {"d1": 2, "d2": 0}, "q2": {"d1": 1}}


def test_load_queries_and_keywords_combines_queries_jsonl_and_id2queries_json(tmp_path):
    (tmp_path / "queries.jsonl").write_text('{"_id": "q1", "text": "condition A"}\n')
    (tmp_path / "id2queries.json").write_text(
        json.dumps({"q1": {"raw": "condition A"}})
    )

    queries, id2queries = load_queries_and_keywords(str(tmp_path))

    assert queries == {"q1": {"_id": "q1", "text": "condition A"}}
    assert id2queries == {"q1": {"raw": "condition A"}}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest infra/tests/test_data_jsonl.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_infra.data'`

- [ ] **Step 4: Implement `jsonl.py`**

Create `infra/src/rag_infra/data/jsonl.py` (relocated unchanged from `Retrieving_stage/poisonrag_experiment/retrieval_utils.py`'s first four functions):

```python
import csv
import json
import os


def load_jsonl(path):
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle]


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_queries_and_keywords(dataset_dir):
    queries = {entry["_id"]: entry for entry in load_jsonl(os.path.join(dataset_dir, "queries.jsonl"))}
    id2queries = json.load(open(os.path.join(dataset_dir, "id2queries.json")))
    return queries, id2queries


def load_qrels(dataset_dir):
    qrels = {}
    path = os.path.join(dataset_dir, "qrels", "test.tsv")
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            doc_id = row["corpus-id"]
            score = int(row["score"])
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest infra/tests/test_data_jsonl.py -v`
Expected: PASS (5 passed)

- [ ] **Step 6: Commit**

```bash
git add infra/src/rag_infra/data/__init__.py infra/src/rag_infra/data/jsonl.py infra/tests/test_data_jsonl.py
git commit -m "feat(infra): add rag_infra.data.jsonl (shared JSON/JSONL/TSV file-I/O helpers)"
```

---

### Task 2: Retrieving_stage adapter — delegate `retrieval_utils.py`'s file-I/O helpers to `rag_infra.data.jsonl`

**Files:**
- Modify: `Retrieving_stage/poisonrag_experiment/retrieval_utils.py`
- Test: `Retrieving_stage/tests/test_retrieval_utils_data_parity.py`

**Interfaces:**
- Consumes: `rag_infra.data.jsonl.{load_jsonl, dump_json, load_qrels, load_queries_and_keywords}` (Task 1).
- Produces: `poisonrag_experiment.retrieval_utils.{load_jsonl, dump_json, load_qrels, load_queries_and_keywords}` remain importable with identical signatures — used unchanged by `Retrieving_stage/poisonrag_experiment/run_poisonrag_experiment.py`'s existing `from poisonrag_experiment.retrieval_utils import (...)` block, and by `retrieval_utils.py`'s own `build_medcpt_corpus_index` function (internal use of `dump_json`).

- [ ] **Step 1: Write the failing parity test**

Create `Retrieving_stage/tests/test_retrieval_utils_data_parity.py`:

```python
from __future__ import annotations

from poisonrag_experiment.retrieval_utils import (
    dump_json,
    load_jsonl,
    load_qrels,
    load_queries_and_keywords,
)
from rag_infra.data.jsonl import dump_json as core_dump_json
from rag_infra.data.jsonl import load_jsonl as core_load_jsonl
from rag_infra.data.jsonl import load_qrels as core_load_qrels
from rag_infra.data.jsonl import load_queries_and_keywords as core_load_queries_and_keywords


def test_retrieval_utils_data_helpers_reexport_rag_infra_exactly():
    assert load_jsonl is core_load_jsonl
    assert dump_json is core_dump_json
    assert load_qrels is core_load_qrels
    assert load_queries_and_keywords is core_load_queries_and_keywords
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Retrieving_stage/tests/test_retrieval_utils_data_parity.py -v`
Expected: FAIL — `assert False` (these are currently `retrieval_utils.py`'s own local functions, not `rag_infra`'s)

- [ ] **Step 3: Replace the four function bodies with an import from `rag_infra.data.jsonl`**

In `Retrieving_stage/poisonrag_experiment/retrieval_utils.py`, make exactly these mechanical edits — do not touch anything else in the file (everything from `get_conditions` onward, including `build_medcpt_corpus_index`'s existing call to `dump_json(doc_ids_path, doc_ids)`, stays byte-for-byte as it is today):

1. Replace lines 1-11 (the current top-of-file, from `import csv` through the last `from transformers import ...` line and the two blank lines that follow it) with:

```python
import json
import os

import faiss
import numpy as np
import torch
import tqdm
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from rag_infra.data.jsonl import dump_json, load_jsonl, load_qrels, load_queries_and_keywords
```

This drops `import csv` (nothing in the file uses `csv` directly once `load_qrels` — the only caller of `csv.DictReader` — is removed in the next step) and keeps `import json`/`import os` (still used by `build_medcpt_corpus_index` for `json.load(...)`/`os.makedirs`/`os.path.join`/`os.path.exists`), adding one new import line for the four relocated functions.

2. Delete the four function definitions that originally followed (`load_jsonl`, `dump_json`, `load_queries_and_keywords`, `load_qrels` — originally lines 14-41 of the file, ending right before `def get_conditions(...)`). After this deletion, the file should go directly from the new import block in step 1 to `def get_conditions(id2queries, qid, query_type):` with a single blank-line gap, matching the file's existing spacing convention between top-level defs.

Read the file first to confirm current line numbers match this description before editing (an earlier task in this plan doesn't touch this file, so line numbers should be unchanged from what's described here, but verify before deleting).

- [ ] **Step 4: Run parity test to verify it passes**

Run: `pytest Retrieving_stage/tests/test_retrieval_utils_data_parity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Run the full Retrieving_stage test suite to confirm nothing broke**

Run: `pytest Retrieving_stage/tests/ -v`
Expected: all tests PASS, including the pre-existing `test_drs_parity.py` and `test_ollama_utils_parity.py` (from Phase 1)

- [ ] **Step 6: Verify the untouched caller still imports correctly**

`Retrieving_stage/poisonrag_experiment/run_poisonrag_experiment.py` is not modified in this task, but confirm its import block still resolves without error (it imports many names from `retrieval_utils`, both moved and unmoved):

Run: `python3 -c "import sys; sys.path.insert(0, 'Retrieving_stage'); from poisonrag_experiment.run_poisonrag_experiment import main"`
Expected: no `ImportError`/`ModuleNotFoundError` (the script's heavier dependencies — Ollama, torch models — aren't invoked by just importing it, so this only proves the import graph is intact, not that the script runs end-to-end)

- [ ] **Step 7: Commit**

```bash
git add Retrieving_stage/poisonrag_experiment/retrieval_utils.py Retrieving_stage/tests/test_retrieval_utils_data_parity.py
git commit -m "refactor(Retrieving_stage): delegate retrieval_utils file-I/O helpers to rag_infra.data.jsonl"
```

---

### Task 3: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none — this task only runs the test surfaces produced by Tasks 1–2.

- [ ] **Step 1: Run every test suite in the repo**

```bash
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest RAG_Setting/tests/ -v
pytest Agent_Setting/tests/ -v
pytest Retrieving_stage/tests/ -v
```

Expected: all PASS. This confirms the narrowed Phase 2 extraction didn't regress Phase 1's work or the original `drs_defense` shared-module work.

- [ ] **Step 2: Grep for any remaining duplicate definitions**

```bash
grep -rn "^def load_jsonl\|^def dump_json\|^def load_qrels\|^def load_queries_and_keywords" Retrieving_stage/poisonrag_experiment
```

Expected: no output — confirms `retrieval_utils.py` no longer defines these four functions locally (only imports them).

- [ ] **Step 3: Report results to the user**

Summarize: which files were created/modified, all test results, and confirm the four file-I/O helpers are now single-sourced in `rag_infra.data.jsonl`. Note explicitly that RAG_Setting's `medqa_loader.py`/`pubmed_loader.py` and Agent_Setting/ReAct's StrategyQA loading remain subproject-local by design (per the narrowed Phase 2 scope decision), not because they were missed. No commit needed for this task (verification only).
