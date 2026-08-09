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

`Retrieving_stage/trialgpt_retrieval/keyword_generation.py`'s `generate_with_ollama`
is a known fourth Ollama-client duplicate (same `urllib` POST to `/api/generate`,
`"format": "json"`, and empty-response check as `json_client.generate_json`) that was
deliberately left unextracted here: it returns raw text instead of parsed JSON,
hardcodes `temperature: 0`, flattens a messages list into `system`/`prompt`, and raises
`RuntimeError` instead of `OllamaError`, so folding it into `json_client` would be a
behavior change. Deferred to a later phase.

## `rag_infra.data`

`jsonl.py` — small JSON/JSONL/TSV file-I/O helpers used by
`Retrieving_stage/poisonrag_experiment`:

- `load_jsonl(path)` / `dump_json(path, payload)` — generic JSONL read and
  JSON write (the latter creates the parent directory via `os.makedirs` if
  needed).
- `load_qrels(dataset_dir)` — parses a BEIR/SIGIR/TREC-style
  `qrels/test.tsv` (`query-id`/`corpus-id`/`score` columns) into a
  `{query_id: {doc_id: score}}` mapping.
- `load_queries_and_keywords(dataset_dir)` — loads a `queries.jsonl` (keyed
  by `_id`) together with its sibling `id2queries.json` cache, the standard
  BEIR/SIGIR/TREC dataset-directory layout used under
  `Retrieving_stage/dataset/{sigir,trec_2021,trec_2022}/`.

`Retrieving_stage/poisonrag_experiment/retrieval_utils.py` is a thin adapter
over this module — do not reimplement this file I/O locally; add it here and
delegate.

`RAG_Setting/src/medrag_repro/utils/io.py` has near-twin functions
(`read_jsonl`, `write_json`) that were deliberately NOT folded into
`jsonl.py`: they differ behaviorally — `Path`-based instead of raw string
paths, explicit `encoding="utf-8"` on every open, `ensure_ascii=False` on
JSON output (vs. `jsonl.py`'s ASCII-escaping default), blank-line skipping
in the JSONL reader, and automatic dataclass `asdict()` conversion when
writing JSONL rows. Deferred to a later phase, same as the
`keyword_generation.py` duplicate above.

## `rag_infra.defenses`

`l2_norm.py` — `l2_norm_score(embeddings)`: row-wise Euclidean (L2) norm of an
embedding matrix (`np.linalg.norm(embeddings, axis=1)`), promoting a 1-D input
to a single row first. Used as a poisoning-detection score: outlier
embeddings tend to have unusually large or small norms relative to the clean
corpus.

Two adapters delegate to it instead of reimplementing the math:

- `RAG_Setting/src/medrag_repro/defense/l2_norm.py`'s `L2NormDetector` —
  encodes texts with the Contriever encoder, calls `l2_norm_score` on the
  resulting `float64` matrix, and fits two-sided quantile thresholds on the
  clean-score distribution.
- `Agent_Setting/ReAct/defense_baselines.py`'s `l2_norm_scores` — converts a
  `torch.Tensor` of embeddings to a NumPy array, calls `l2_norm_score`, and
  converts the result back to a `torch.Tensor` on the original device.

Two other baseline defenses live alongside `l2_norm` in both subprojects but
were evaluated and NOT extracted here:

- `l2_distance` — `RAG_Setting/src/medrag_repro/defense/l2_distance.py`'s
  `L2DistanceDetector` scores each text by its distance to the *centroid* of
  the clean embeddings (`np.linalg.norm(X - self.centroid, axis=1)`), while
  `Agent_Setting/ReAct/defense_baselines.py`'s `l2_distance_scores` scores by
  distance to the *nearest* clean-reference embedding
  (`torch.cdist(embeddings, clean_reference).min(dim=1)`). These are
  different formulas (centroid distance vs. nearest-neighbor distance), not
  duplicates, so there is nothing to fold into a shared function.
- `perplexity` — `RAG_Setting/src/medrag_repro/defense/perplexity.py`'s
  `PerplexityDetector` and `Agent_Setting/ReAct/defense_baselines.py`'s
  `PerplexityScorer` both compute the same underlying score
  (`exp(causal-LM loss)` via a Hugging Face `AutoModelForCausalLM` forward
  pass with `labels=input_ids`), but extracting it would pull `torch` and
  `transformers` into `rag_infra`'s dependencies. Deferred alongside the
  `infra/retrieval/` embedding-model code (Phase 3), pending a decision on
  adding `torch`+`transformers` as hard `rag_infra` dependencies.
