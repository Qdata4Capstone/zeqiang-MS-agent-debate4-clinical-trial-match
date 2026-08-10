# rag_infra

Shared execution infrastructure used across `use-cases/trial_retrieval/`, `use-cases/medqa_rag/`,
and `use-cases/strategyqa_agent/` — extracted so it stops being duplicated per subproject,
following the same pattern already used for `drs_defense/`.

## Code structure

```
infra/
  src/rag_infra/
    llm/
      client.py       # OpenAI-compatible chat completion (chat_completion, load_openai_client)
      ollama.py        # native Ollama /api/generate completion (ollama_generate, ollama_completion)
      json_client.py   # native Ollama /api/generate JSON-mode completion (generate_json, OllamaError)
    data/
      jsonl.py          # JSONL/JSON/TSV I/O (load_jsonl, dump_json, load_qrels, load_queries_and_keywords)
  tests/                 # pytest suite for rag_infra.llm and rag_infra.data.jsonl
```

## Install

```bash
pip install -e ./infra
```

## Quick start

Requires a running Ollama server with the target model pulled:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

```python
from rag_infra.llm.client import load_openai_client, chat_completion

# load_openai_client() reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment
client = load_openai_client()
answer = chat_completion(
    client,
    model="qwen2.5:7b-instruct",
    system="You are a helpful assistant.",
    user="Say hello in one word.",
)

from rag_infra.llm.json_client import generate_json

parsed = generate_json(
    model="qwen2.5:7b-instruct",
    prompt='Return {"greeting": "hello"} as JSON.',
)

from rag_infra.data.jsonl import load_jsonl

rows = load_jsonl("path/to/file.jsonl")
```

`chat_completion` needs `OPENAI_BASE_URL=http://127.0.0.1:11434/v1` and `OPENAI_API_KEY=ollama` set (or any other OpenAI-compatible endpoint); `generate_json` talks to Ollama's native `/api/generate` directly and needs no environment variables (defaults to `http://localhost:11434`).

## `rag_infra.llm`

Three LLM-client implementations, each used differently by the subprojects:

- `client.py` — OpenAI-compatible chat completion (`chat_completion`,
  `load_openai_client`). Used by `use-cases/medqa_rag` for generation and evaluation
  against an Ollama-served OpenAI-compatible endpoint.
- `ollama.py` — native Ollama `/api/generate` text completion
  (`ollama_generate`, `ollama_completion`), including an OpenAI-completions-style
  logprobs shim. Used by `use-cases/strategyqa_agent/ReAct` for stepwise ReAct prompting
  with stop sequences.
- `json_client.py` — native Ollama `/api/generate` with forced JSON output
  (`generate_json`). Used by `use-cases/trial_retrieval/poisonrag_experiment` for
  structured poison generation.

The remaining per-project client files (`use-cases/medqa_rag/src/medrag_repro/llm/client.py`,
`use-cases/strategyqa_agent/ReAct/ollama_client.py`, `use-cases/trial_retrieval/poisonrag_experiment/ollama_utils.py`)
are thin re-export adapters over this package that preserve each subproject's
existing call signatures.

`use-cases/trial_retrieval/trialgpt_retrieval/keyword_generation.py`'s `generate_with_ollama`
is a known fourth Ollama-client duplicate (same `urllib` POST to `/api/generate`,
`"format": "json"`, and empty-response check as `json_client.generate_json`) that was
deliberately left unextracted here: it returns raw text instead of parsed JSON,
hardcodes `temperature: 0`, flattens a messages list into `system`/`prompt`, and raises
`RuntimeError` instead of `OllamaError`, so folding it into `json_client` would be a
behavior change. Deferred to a later phase.

## `rag_infra.data`

`jsonl.py` — small JSON/JSONL/TSV file-I/O helpers used by
`use-cases/trial_retrieval/poisonrag_experiment`:

- `load_jsonl(path)` / `dump_json(path, payload)` — generic JSONL read and
  JSON write (the latter creates the parent directory via `os.makedirs` if
  needed).
- `load_qrels(dataset_dir)` — parses a BEIR/SIGIR/TREC-style
  `qrels/test.tsv` (`query-id`/`corpus-id`/`score` columns) into a
  `{query_id: {doc_id: score}}` mapping.
- `load_queries_and_keywords(dataset_dir)` — loads a `queries.jsonl` (keyed
  by `_id`) together with its sibling `id2queries.json` cache, the standard
  BEIR/SIGIR/TREC dataset-directory layout used under
  `use-cases/trial_retrieval/dataset/{sigir,trec_2021,trec_2022}/`.

`use-cases/trial_retrieval/poisonrag_experiment/retrieval_utils.py` is a thin adapter
over this module — do not reimplement this file I/O locally; add it here and
delegate.

`use-cases/medqa_rag/src/medrag_repro/utils/io.py` has near-twin functions
(`read_jsonl`, `write_json`) that were deliberately NOT folded into
`jsonl.py`: they differ behaviorally — `Path`-based instead of raw string
paths, explicit `encoding="utf-8"` on every open, `ensure_ascii=False` on
JSON output (vs. `jsonl.py`'s ASCII-escaping default), blank-line skipping
in the JSONL reader, and automatic dataclass `asdict()` conversion when
writing JSONL rows. Deferred to a later phase, same as the
`keyword_generation.py` duplicate above.
