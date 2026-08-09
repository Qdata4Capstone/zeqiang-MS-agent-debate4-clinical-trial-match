# infra/llm/ Extraction (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the three duplicated LLM-client implementations (RAG_Setting's OpenAI-compatible chat client, Agent_Setting's native-Ollama completion client, Retrieving_stage's Ollama JSON-generation client) into one new shared, pip-installable package, following the exact `drs_defense` precedent (shared implementation + thin per-subproject re-export adapter + identity-based parity test).

**Architecture:** A new top-level `infra/` package (dist name `rag-infra`, import name `rag_infra`, src-layout, `requires-python = ">=3.9"` to match the strictest existing constraint) holds three modules under `rag_infra.llm`: `client.py` (OpenAI-compatible chat completion), `ollama.py` (native Ollama `/api/generate` text completion, used by the ReAct agent's stepwise prompting), and `json_client.py` (native Ollama `/api/generate` with forced JSON output, used by poison generation). Each subproject's existing client file becomes a two-line re-export of the shared module, so every existing caller (6 in RAG_Setting, 2 in Agent_Setting, 1 in Retrieving_stage) keeps working with zero changes to call sites.

**Tech Stack:** Python, `openai` SDK, `requests`, stdlib `urllib`, `pytest`, `unittest.mock`.

## Global Constraints

- `rag_infra` must stay importable under both Python 3.9 (Agent_Setting's pinned conda env) and Python 3.10 (RAG_Setting's conda env) — no syntax newer than 3.9 allows (this only binds `llm/`, but keep it in mind since later phases add retrieval code to the same package).
- No changes to call-site behavior: every existing caller of `medrag_repro.llm.client`, `ReAct.ollama_client`, and `poisonrag_experiment.ollama_utils` must continue to work with unmodified import statements.
- No new attack or defense logic, no renaming of `Retrieving_stage/`, `RAG_Setting/`, `Agent_Setting/` top-level directories (per the design spec's non-goals).
- Follow the `drs_defense` precedent exactly: shared implementation + thin adapter + parity test proving the adapter is the shared implementation (not a reimplementation of it).
- Run all commands from the `safematch_v3` worktree root: `/Users/qiyanjun/Code/Public/zeqiang-MS-agent-debate4-clinical-trial-match/.worktrees/safematch_v3`. Use whatever `python3`/`pip` the shell already resolves to — this environment already has `drs-defense` and `medrag-repro` installed editable (confirmed via `pip show`), so it's the same environment used for cross-subproject test verification throughout this repo's `drs_defense` work.

---

### Task 1: Scaffold `rag-infra` package + `rag_infra.llm.client` (chat completion)

**Files:**
- Create: `infra/pyproject.toml`
- Create: `infra/README.md`
- Create: `infra/src/rag_infra/__init__.py`
- Create: `infra/src/rag_infra/llm/__init__.py`
- Create: `infra/src/rag_infra/llm/client.py`
- Test: `infra/tests/test_client.py`

**Interfaces:**
- Produces: `rag_infra.llm.client.load_openai_client() -> openai.OpenAI`
- Produces: `rag_infra.llm.client.chat_completion(client: openai.OpenAI, model: str, system: str, user: str, temperature: float = 0.2, max_tokens: int = 512) -> str`
- Produces: the `rag_infra` package installed editable in the active environment (subsequent tasks assume `import rag_infra...` resolves).

- [ ] **Step 1: Create the package skeleton**

Create `infra/pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "rag-infra"
version = "0.1.0"
description = "Shared LLM-client infrastructure (OpenAI-compatible chat completion and native Ollama completion clients) used across Retrieving_stage, RAG_Setting, and Agent_Setting."
readme = "README.md"
requires-python = ">=3.9"
dependencies = [
  "openai>=1.30.0",
  "requests>=2.31",
]

[project.optional-dependencies]
dev = ["pytest>=7"]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
```

Create `infra/README.md`:

```markdown
# rag_infra

Shared execution infrastructure used across `Retrieving_stage/`, `RAG_Setting/`,
and `Agent_Setting/` — extracted so it stops being duplicated per subproject,
following the same pattern already used for `drs_defense/`.

## `rag_infra.llm`

Three LLM-client implementations, each used differently by the subprojects:

- `client.py` — OpenAI-compatible chat completion (`chat_completion`,
  `load_openai_client`). Used by `RAG_Setting` for generation and evaluation
  against an Ollama-served OpenAI-compatible endpoint.
- `ollama.py` — native Ollama `/api/generate` text completion
  (`ollama_generate`, `ollama_completion`), including an OpenAI-completions-style
  logprobs shim. Used by `Agent_Setting/ReAct` for stepwise ReAct prompting
  with stop sequences.
- `json_client.py` — native Ollama `/api/generate` with forced JSON output
  (`generate_json`). Used by `Retrieving_stage/poisonrag_experiment` for
  structured poison generation.

The remaining per-project client files (`RAG_Setting/src/medrag_repro/llm/client.py`,
`Agent_Setting/ReAct/ollama_client.py`, `Retrieving_stage/poisonrag_experiment/ollama_utils.py`)
are thin re-export adapters over this package that preserve each subproject's
existing call signatures.
```

Create `infra/src/rag_infra/__init__.py` (empty file).

Create `infra/src/rag_infra/llm/__init__.py` (empty file).

- [ ] **Step 2: Install the package editable**

Run: `pip install -e infra`
Expected: `Successfully installed rag-infra-0.1.0`

- [ ] **Step 3: Write the failing test**

Create `infra/tests/test_client.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock

from rag_infra.llm.client import chat_completion, load_openai_client


def test_load_openai_client_defaults_ollama_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = load_openai_client()

    assert client.api_key == "ollama"
    assert "localhost:11434" in str(client.base_url)


def test_load_openai_client_requires_api_key_when_not_ollama(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        load_openai_client()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)


def test_chat_completion_sends_system_and_user_messages_and_returns_content():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello world"))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    result = chat_completion(
        fake_client,
        model="qwen2.5:7b-instruct",
        system="You are a helpful assistant.",
        user="Say hi.",
        temperature=0.5,
        max_tokens=64,
    )

    assert result == "hello world"
    fake_client.chat.completions.create.assert_called_once_with(
        model="qwen2.5:7b-instruct",
        temperature=0.5,
        max_tokens=64,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi."},
        ],
    )


def test_chat_completion_returns_empty_string_when_content_is_none():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=None))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    result = chat_completion(fake_client, model="m", system="s", user="u")

    assert result == ""
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest infra/tests/test_client.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_infra.llm.client'`

- [ ] **Step 5: Implement `client.py`**

Create `infra/src/rag_infra/llm/client.py` (relocated unchanged from `RAG_Setting/src/medrag_repro/llm/client.py`):

```python
from __future__ import annotations

import os
from openai import OpenAI


def load_openai_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    # Ollama local mode: default to a placeholder API key.
    if base_url and "localhost:11434" in base_url and not api_key:
        api_key = "ollama"

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    return OpenAI(api_key=api_key, base_url=base_url)


def chat_completion(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
```

Note: the original file's comment was in Chinese (`# Ollama 本地模式自动兜底`); translate it as shown above — behavior is unchanged, only the comment language, so this doesn't require a parity test to cover.

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest infra/tests/test_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 7: Commit**

```bash
git add infra/pyproject.toml infra/README.md infra/src/rag_infra/__init__.py infra/src/rag_infra/llm/__init__.py infra/src/rag_infra/llm/client.py infra/tests/test_client.py
git commit -m "feat(infra): add rag_infra shared package with llm.client (chat completion)"
```

---

### Task 2: `rag_infra.llm.ollama` (native Ollama completion, ReAct-style)

**Files:**
- Create: `infra/src/rag_infra/llm/ollama.py`
- Test: `infra/tests/test_ollama.py`

**Interfaces:**
- Consumes: nothing from Task 1 (independent module in the same package).
- Produces: `rag_infra.llm.ollama.DEFAULT_OLLAMA_BASE_URL: str`, `DEFAULT_OLLAMA_MODEL: str`, `DEFAULT_TIMEOUT: int`
- Produces: `rag_infra.llm.ollama.ollama_generate(prompt, system_prompt=None, stop=None, temperature=0.0, max_tokens=256, model=None, timeout=DEFAULT_TIMEOUT) -> str`
- Produces: `rag_infra.llm.ollama.ollama_completion(prompt, stop=None, return_probs=False, system_prompt=None, temperature=0.0, max_tokens=256, model=None, timeout=DEFAULT_TIMEOUT) -> str | dict`
- Produces: `rag_infra.llm.ollama._tokenize_for_compatibility(text: str) -> list[str]`

- [ ] **Step 1: Write the failing test**

Create `infra/tests/test_ollama.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_infra.llm import ollama


def test_ollama_generate_posts_expected_payload_and_returns_response_text():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "42"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response) as mock_post:
        result = ollama.ollama_generate(
            prompt="What is 6*7?",
            system_prompt="Answer tersely.",
            stop=["\n"],
            temperature=0.1,
            max_tokens=16,
            model="qwen2.5:7b-instruct",
        )

    assert result == "42"
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/api/generate")
    assert kwargs["json"]["model"] == "qwen2.5:7b-instruct"
    assert kwargs["json"]["prompt"] == "What is 6*7?"
    assert kwargs["json"]["system"] == "Answer tersely."
    assert kwargs["json"]["options"]["stop"] == ["\n"]
    assert kwargs["json"]["options"]["temperature"] == 0.1
    assert kwargs["json"]["options"]["num_predict"] == 16


def test_ollama_generate_omits_system_and_stop_when_not_provided():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "ok"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response) as mock_post:
        ollama.ollama_generate(prompt="hi")

    _, kwargs = mock_post.call_args
    assert "system" not in kwargs["json"]
    assert "stop" not in kwargs["json"]["options"]


def test_ollama_completion_without_probs_returns_plain_text():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "plain answer"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response):
        result = ollama.ollama_completion(prompt="hi", return_probs=False)

    assert result == "plain answer"


def test_ollama_completion_with_probs_returns_logprobs_shim():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "two words"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response):
        result = ollama.ollama_completion(prompt="hi", return_probs=True)

    assert result["text"] == "two words"
    assert result["logprobs"]["tokens"] == ollama._tokenize_for_compatibility("two words")
    assert result["logprobs"]["token_logprobs"] == [0.0, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest infra/tests/test_ollama.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_infra.llm.ollama'`

- [ ] **Step 3: Implement `ollama.py`**

Create `infra/src/rag_infra/llm/ollama.py` (relocated unchanged from `Agent_Setting/ReAct/ollama_client.py`):

```python
import os
import re

import requests


DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))


def _tokenize_for_compatibility(text):
    return re.findall(r"\n| [A-Za-z]+|[A-Za-z]+| ?\d+| ?\[[^\]]*|\]| ?[:.,!?;()_-]", text)


def ollama_generate(
    prompt,
    system_prompt=None,
    stop=None,
    temperature=0.0,
    max_tokens=256,
    model=None,
    timeout=DEFAULT_TIMEOUT,
):
    payload = {
        "model": model or DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if system_prompt:
        payload["system"] = system_prompt
    if stop:
        payload["options"]["stop"] = stop

    response = requests.post(
        f"{DEFAULT_OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]


def ollama_completion(
    prompt,
    stop=None,
    return_probs=False,
    system_prompt=None,
    temperature=0.0,
    max_tokens=256,
    model=None,
    timeout=DEFAULT_TIMEOUT,
):
    text = ollama_generate(
        prompt=prompt,
        system_prompt=system_prompt,
        stop=stop,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        timeout=timeout,
    )

    if not return_probs:
        return text

    tokens = _tokenize_for_compatibility(text)
    if not tokens:
        tokens = [text]

    return {
        "text": text,
        "logprobs": {
            "tokens": tokens,
            "token_logprobs": [0.0 for _ in tokens],
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest infra/tests/test_ollama.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add infra/src/rag_infra/llm/ollama.py infra/tests/test_ollama.py
git commit -m "feat(infra): add rag_infra.llm.ollama (native Ollama completion client)"
```

---

### Task 3: `rag_infra.llm.json_client` (Ollama JSON generation)

**Files:**
- Create: `infra/src/rag_infra/llm/json_client.py`
- Test: `infra/tests/test_json_client.py`

**Interfaces:**
- Consumes: nothing from Tasks 1–2 (independent module in the same package).
- Produces: `rag_infra.llm.json_client.OllamaError` (exception class)
- Produces: `rag_infra.llm.json_client.generate_json(model, prompt, system=None, base_url="http://localhost:11434", temperature=0.7, timeout=300) -> dict`

- [ ] **Step 1: Write the failing test**

Create `infra/tests/test_json_client.py`:

```python
from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from rag_infra.llm.json_client import OllamaError, generate_json


def _fake_urlopen_response(body_dict):
    fake = MagicMock()
    fake.read.return_value = json.dumps(body_dict).encode("utf-8")
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = False
    return fake


def test_generate_json_parses_response_field_as_json():
    fake_response = _fake_urlopen_response({"response": '{"keywords": ["a", "b"]}'})

    with patch("rag_infra.llm.json_client.urllib.request.urlopen", return_value=fake_response):
        result = generate_json(model="qwen2.5:7b-instruct", prompt="extract keywords")

    assert result == {"keywords": ["a", "b"]}


def test_generate_json_raises_ollama_error_on_empty_response():
    fake_response = _fake_urlopen_response({"response": ""})

    with patch("rag_infra.llm.json_client.urllib.request.urlopen", return_value=fake_response):
        with pytest.raises(OllamaError, match="empty response"):
            generate_json(model="m", prompt="p")


def test_generate_json_raises_ollama_error_on_invalid_json():
    fake_response = _fake_urlopen_response({"response": "not json"})

    with patch("rag_infra.llm.json_client.urllib.request.urlopen", return_value=fake_response):
        with pytest.raises(OllamaError, match="not valid JSON"):
            generate_json(model="m", prompt="p")


def test_generate_json_raises_ollama_error_on_connection_failure():
    with patch(
        "rag_infra.llm.json_client.urllib.request.urlopen",
        side_effect=urllib.error.URLError("boom"),
    ):
        with pytest.raises(OllamaError, match="Failed to connect"):
            generate_json(model="m", prompt="p", base_url="http://localhost:11434")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest infra/tests/test_json_client.py -v`
Expected: FAIL/ERROR — `ModuleNotFoundError: No module named 'rag_infra.llm.json_client'`

- [ ] **Step 3: Implement `json_client.py`**

Create `infra/src/rag_infra/llm/json_client.py` (relocated unchanged from `Retrieving_stage/poisonrag_experiment/ollama_utils.py`):

```python
import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when an Ollama request fails."""


def generate_json(
    model,
    prompt,
    system=None,
    base_url="http://localhost:11434",
    temperature=0.7,
    timeout=300,
):
    """Call Ollama and return a parsed JSON object from the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    if system:
        payload["system"] = system

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Failed to connect to Ollama at {base_url}: {exc}"
        ) from exc

    text = body.get("response", "").strip()
    if not text:
        raise OllamaError("Ollama returned an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OllamaError(
            "Ollama response was not valid JSON. "
            f"Raw response: {text[:500]}"
        ) from exc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest infra/tests/test_json_client.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add infra/src/rag_infra/llm/json_client.py infra/tests/test_json_client.py
git commit -m "feat(infra): add rag_infra.llm.json_client (Ollama JSON generation client)"
```

---

### Task 4: RAG_Setting adapter over `rag_infra.llm.client`

**Files:**
- Modify: `RAG_Setting/src/medrag_repro/llm/client.py`
- Modify: `RAG_Setting/requirements.txt`
- Test: `RAG_Setting/tests/test_llm_client_parity.py`

**Interfaces:**
- Consumes: `rag_infra.llm.client.chat_completion`, `rag_infra.llm.client.load_openai_client` (Task 1).
- Produces: `medrag_repro.llm.client.chat_completion` and `medrag_repro.llm.client.load_openai_client` remain importable with identical signatures (used unchanged by `RAG_Setting/src/medrag_repro/evaluation/rag_eval.py`, `RAG_Setting/src/medrag_repro/attacks/poisonedrag_blackbox.py`, and `RAG_Setting/scripts/{run_defense,generate_poison,run_drs,eval_attack}.py`).

- [ ] **Step 1: Write the failing parity test**

Create `RAG_Setting/tests/test_llm_client_parity.py`:

```python
from __future__ import annotations

from medrag_repro.llm.client import chat_completion, load_openai_client
from rag_infra.llm.client import chat_completion as core_chat_completion
from rag_infra.llm.client import load_openai_client as core_load_openai_client


def test_medrag_repro_llm_client_reexports_rag_infra_exactly():
    assert chat_completion is core_chat_completion
    assert load_openai_client is core_load_openai_client
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest RAG_Setting/tests/test_llm_client_parity.py -v`
Expected: FAIL — `assert False` (`chat_completion` is currently `medrag_repro`'s own local function, not `rag_infra`'s)

- [ ] **Step 3: Replace the file with a thin re-export**

Replace the full contents of `RAG_Setting/src/medrag_repro/llm/client.py` with:

```python
from __future__ import annotations

from rag_infra.llm.client import chat_completion, load_openai_client

__all__ = ["chat_completion", "load_openai_client"]
```

- [ ] **Step 4: Add the `rag-infra` dependency**

Modify `RAG_Setting/requirements.txt` (currently `-e .` / `-e ../drs_defense` / `pytest`) to add the new editable dependency:

```
-e .
-e ../drs_defense
-e ../infra
pytest
```

- [ ] **Step 5: Run parity test to verify it passes**

Run: `pytest RAG_Setting/tests/test_llm_client_parity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full RAG_Setting test suite to confirm nothing broke**

Run: `pytest RAG_Setting/tests/ -v`
Expected: all tests PASS, including the pre-existing `test_drs_detector_parity.py`

- [ ] **Step 7: Commit**

```bash
git add RAG_Setting/src/medrag_repro/llm/client.py RAG_Setting/requirements.txt RAG_Setting/tests/test_llm_client_parity.py
git commit -m "refactor(RAG_Setting): delegate llm.client to rag_infra, keep call signatures"
```

---

### Task 5: Agent_Setting adapter over `rag_infra.llm.ollama`

**Files:**
- Modify: `Agent_Setting/ReAct/ollama_client.py`
- Modify: `Agent_Setting/environment.yml`
- Test: `Agent_Setting/tests/test_ollama_client_parity.py`

**Interfaces:**
- Consumes: `rag_infra.llm.ollama.{ollama_generate, ollama_completion, DEFAULT_OLLAMA_BASE_URL, DEFAULT_OLLAMA_MODEL, DEFAULT_TIMEOUT}` (Task 2).
- Produces: `ReAct.ollama_client.ollama_completion` and `ReAct.ollama_client.ollama_generate` remain importable with identical signatures (used unchanged by `Agent_Setting/ReAct/run_strategyqa_inference.py` and `Agent_Setting/algo/utils.py`).

- [ ] **Step 1: Write the failing parity test**

Create `Agent_Setting/tests/test_ollama_client_parity.py`:

```python
from __future__ import annotations

from rag_infra.llm.ollama import ollama_completion as core_ollama_completion
from rag_infra.llm.ollama import ollama_generate as core_ollama_generate
from ReAct.ollama_client import ollama_completion, ollama_generate


def test_react_ollama_client_reexports_rag_infra_exactly():
    assert ollama_generate is core_ollama_generate
    assert ollama_completion is core_ollama_completion
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Agent_Setting/tests/test_ollama_client_parity.py -v`
Expected: FAIL — `assert False` (`ollama_generate` is currently `ReAct`'s own local function, not `rag_infra`'s)

- [ ] **Step 3: Replace the file with a thin re-export**

Replace the full contents of `Agent_Setting/ReAct/ollama_client.py` with:

```python
from rag_infra.llm.ollama import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TIMEOUT,
    ollama_completion,
    ollama_generate,
)

__all__ = [
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_TIMEOUT",
    "ollama_completion",
    "ollama_generate",
]
```

- [ ] **Step 4: Add the `rag-infra` dependency**

Modify `Agent_Setting/environment.yml`: in the `pip:` list, add `- -e ../infra` next to the existing `- -e ../drs_defense` line (line 57):

```yaml
    - -e ../drs_defense
    - -e ../infra
    - pytest
```

- [ ] **Step 5: Run parity test to verify it passes**

Run: `pytest Agent_Setting/tests/test_ollama_client_parity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full Agent_Setting test suite to confirm nothing broke**

Run: `pytest Agent_Setting/tests/ -v`
Expected: all tests PASS, including the pre-existing `test_drs_parity.py`

- [ ] **Step 7: Commit**

```bash
git add Agent_Setting/ReAct/ollama_client.py Agent_Setting/environment.yml Agent_Setting/tests/test_ollama_client_parity.py
git commit -m "refactor(Agent_Setting): delegate ReAct.ollama_client to rag_infra, keep call signatures"
```

---

### Task 6: Retrieving_stage adapter over `rag_infra.llm.json_client`

**Files:**
- Modify: `Retrieving_stage/poisonrag_experiment/ollama_utils.py`
- Modify: `Retrieving_stage/requirements.txt`
- Test: `Retrieving_stage/tests/test_ollama_utils_parity.py`

**Interfaces:**
- Consumes: `rag_infra.llm.json_client.{generate_json, OllamaError}` (Task 3).
- Produces: `poisonrag_experiment.ollama_utils.generate_json` and `poisonrag_experiment.ollama_utils.OllamaError` remain importable with identical signatures (used unchanged by `Retrieving_stage/poisonrag_experiment/run_poisonrag_experiment.py`).

- [ ] **Step 1: Write the failing parity test**

Create `Retrieving_stage/tests/test_ollama_utils_parity.py`:

```python
from __future__ import annotations

from poisonrag_experiment.ollama_utils import OllamaError, generate_json
from rag_infra.llm.json_client import OllamaError as CoreOllamaError
from rag_infra.llm.json_client import generate_json as core_generate_json


def test_poisonrag_ollama_utils_reexports_rag_infra_exactly():
    assert generate_json is core_generate_json
    assert OllamaError is CoreOllamaError
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest Retrieving_stage/tests/test_ollama_utils_parity.py -v`
Expected: FAIL — `assert False` (`generate_json` is currently `poisonrag_experiment`'s own local function, not `rag_infra`'s)

- [ ] **Step 3: Replace the file with a thin re-export**

Replace the full contents of `Retrieving_stage/poisonrag_experiment/ollama_utils.py` with:

```python
from rag_infra.llm.json_client import OllamaError, generate_json

__all__ = ["OllamaError", "generate_json"]
```

- [ ] **Step 4: Add the `rag-infra` dependency**

Modify `Retrieving_stage/requirements.txt` (currently ends with `-e ../drs_defense` / `pytest`) to add the new editable dependency:

```
-e ../drs_defense
-e ../infra
pytest
```

(keep all the pinned package lines above those two unchanged)

- [ ] **Step 5: Run parity test to verify it passes**

Run: `pytest Retrieving_stage/tests/test_ollama_utils_parity.py -v`
Expected: PASS (1 passed)

- [ ] **Step 6: Run the full Retrieving_stage test suite to confirm nothing broke**

Run: `pytest Retrieving_stage/tests/ -v`
Expected: all tests PASS, including the pre-existing `test_drs_parity.py`

- [ ] **Step 7: Commit**

```bash
git add Retrieving_stage/poisonrag_experiment/ollama_utils.py Retrieving_stage/requirements.txt Retrieving_stage/tests/test_ollama_utils_parity.py
git commit -m "refactor(Retrieving_stage): delegate poisonrag ollama_utils to rag_infra, keep call signatures"
```

---

### Task 7: Full-repo verification

**Files:** none (verification only, no code changes).

**Interfaces:** none — this task only runs the test surfaces produced by Tasks 1–6.

- [ ] **Step 1: Run every test suite in the repo**

```bash
pytest infra/tests/ -v
pytest drs_defense/tests/ -v
pytest RAG_Setting/tests/ -v
pytest Agent_Setting/tests/ -v
pytest Retrieving_stage/tests/ -v
```

Expected: all PASS. `drs_defense/tests/` passing confirms this phase didn't regress the earlier DRS shared-module work; the three parity suites plus `infra/tests/` confirm the new `rag_infra.llm` package and every adapter are correct.

- [ ] **Step 2: Grep for any remaining direct references to the old local implementations**

```bash
grep -rn "def chat_completion\|def load_openai_client" RAG_Setting/src
grep -rn "def ollama_generate\|def ollama_completion" Agent_Setting/ReAct
grep -rn "def generate_json" Retrieving_stage/poisonrag_experiment
```

Expected: no output — confirms the three original function *definitions* no longer exist outside `infra/src/rag_infra/llm/`, i.e. the adapters are pure re-exports, not parallel copies.

- [ ] **Step 3: Report results to the user**

Summarize: which files were created/modified, all test results, and confirm the three original call-site duplications are now single-sourced in `rag_infra.llm`. No commit needed for this task (verification only).
