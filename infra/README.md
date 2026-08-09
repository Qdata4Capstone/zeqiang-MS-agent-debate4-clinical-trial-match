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
