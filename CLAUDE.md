# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository overview

This is a research repo for clinical-trial matching and retrieval-augmented generation (RAG) robustness experiments. It contains **three independent Python subprojects** (plus two small shared libraries, `drs_defense/` and `infra/`), each with its own environment, dependencies, and entry points — there is no root-level build system, package manager, or test suite tying them together. Treat each directory below as its own project when working inside it.

- `Retrieving_stage/` — TrialGPT-style clinical trial retrieval (keyword generation + hybrid BM25/MedCPT fusion retrieval) plus a corpus-poisoning attack/defense experiment (`poisonrag_experiment/`).
- `RAG_Setting/` — reproduction of PoisonedRAG black-box knowledge poisoning attacks on a medical QA RAG pipeline (MedQA-US + PubMed + Contriever), with DRS and baseline defenses. Installable as the `medrag-repro` package (`src/medrag_repro/`).
- `Agent_Setting/` — adversarial trigger optimization against dense retrievers (AgentPoison-style) plus a ReAct agent (StrategyQA) with DRS defense and baseline comparisons.
- `drs_defense/` — shared, pip-installable reference implementation of the DRS (Directional Relative Shifts) poisoning defense (paper Algorithm 1 & Eq. 3, https://openreview.net/pdf?id=2aL6gcFX7q). `Agent_Setting/ReAct/drs.py`, `RAG_Setting/src/medrag_repro/defense/drs.py`, and `Retrieving_stage/poisonrag_experiment/drs.py` are thin adapters over this module — do not reimplement DRS math locally; add it here and delegate.
- `infra/` — shared, pip-installable `rag_infra` package holding LLM-client infrastructure: OpenAI-compatible chat completion (`rag_infra.llm.client`), native Ollama completion (`rag_infra.llm.ollama`), and Ollama JSON generation (`rag_infra.llm.json_client`). `RAG_Setting/src/medrag_repro/llm/client.py`, `Agent_Setting/ReAct/ollama_client.py`, and `Retrieving_stage/poisonrag_experiment/ollama_utils.py` are thin adapters over this module — do not reimplement LLM-client code locally; add it here and delegate. It also holds `rag_infra.data.jsonl`: JSON/JSONL/TSV dataset file-I/O helpers (`load_jsonl`, `dump_json`, `load_qrels`, `load_queries_and_keywords`) used by `Retrieving_stage/poisonrag_experiment/retrieval_utils.py` — do not reimplement this file I/O locally either; add it here and delegate. It also holds `rag_infra.defenses.l2_norm`: `l2_norm_score`, row-wise L2-norm poisoning-detection scoring, used by `RAG_Setting/src/medrag_repro/defense/l2_norm.py`'s `L2NormDetector` and `Agent_Setting/ReAct/defense_baselines.py`'s `l2_norm_scores` — do not reimplement this scoring math locally either; add it here and delegate.

A root-level `src/` directory (Proposer/Skeptic agents, ranking, eval) existed previously but was deleted (`3616c88 Delete src directory`) and no longer exists — the root `README.md` now reflects this.

All three subprojects share a common LLM backend: **Ollama running `qwen2.5:7b-instruct` locally** (`ollama serve` + `ollama pull qwen2.5:7b-instruct`), used both for generation/answering and for adversarial content generation. Start Ollama before running any script that calls an LLM.

## Working in each subproject

### Retrieving_stage/

Setup: `pip install -r requirements.txt` (from `Retrieving_stage/`).

Trial corpora/queries live under `dataset/{sigir,trec_2021,trec_2022}/` (qrels, queries, corpus, cached `id2queries.json`, `retrieved_trials.json`). The SIGIR corpus is checked in; TREC 2021/2022 corpora must be downloaded (see `README.md`) before running retrieval on those.

- `trialgpt_retrieval/keyword_generation.py` — generates patient keywords via Ollama: `python trialgpt_retrieval/keyword_generation.py <corpus> <model>` where `<corpus>` is `sigir`, `trec_2021`, or `trec_2022`.
- `trialgpt_retrieval/hybrid_fusion_retrieval.py` — BM25 + MedCPT hybrid retrieval using reciprocal-rank fusion over the generated/cached keywords.
- `poisonrag_experiment/` — standalone corpus-poisoning experiment reusing the same retrieval setup: generates synthetic malicious trials with Ollama, injects them into the corpus, and compares `recall@{50,100,200}` before/after poisoning with optional DRS filtering. Run as a module from `Retrieving_stage/`: `python -m poisonrag_experiment.run_poisonrag_experiment --corpus sigir --query_type gpt-4-turbo --num_targets 20 --poisons_per_patient 3 --ollama_model qwen2.5:7b-instruct --output_dir results/poisonrag_sigir`. Key files: `retrieval_utils.py` (retrieval/corpus handling), `drs.py` (defense), `ollama_utils.py` (LLM calls for poison generation).

### RAG_Setting/

Setup (from `RAG_Setting/`):
```bash
conda create -n medrag python=3.10 -y && conda activate medrag
pip install -r requirements.txt   # installs the package itself via `-e .`
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
```

Everything is driven by a single YAML config, `configs/minimal_medqaus_pubmed_contriever.yaml`, which defines paths, dataset sizes, retriever settings (Contriever, top_k=5), PoisonedRAG attack params, LLM eval settings, and DRS/baseline defense params. Scripts under `scripts/` are run in sequence, each taking `--config configs/...yaml`:

1. `scripts/prepare_data.py` — fetch/clean MedQA-US + PubMed abstracts from Hugging Face.
2. `scripts/build_index.py` — build the Contriever corpus index.
3. `scripts/generate_poison.py` — generate PoisonedRAG black-box poison docs.
4. `scripts/eval_attack.py` — evaluate attack success.
5. `scripts/run_drs.py` — fit/run the DRS defense.
6. `scripts/run_defense.py --method {drs,l2_norm,l2_distance,perplexity}` — compare defense methods.

Package layout (`src/medrag_repro/`): `retriever/` (Contriever + index), `attacks/` (`poisonedrag_blackbox.py`), `defense/` (`drs.py`, `l2_norm.py`, `l2_distance.py`, `perplexity.py`, plus a `common.py` shared by defenses), `data/` (`pubmed_loader.py`, `medqa_loader.py`), `llm/` (`client.py`, `prompts.py` — the Ollama/OpenAI-compatible LLM client used across attack generation, answering, and eval), `evaluation/rag_eval.py`, and `config.py` (`load_config()` for the YAML above).

### Agent_Setting/

Setup: `conda env create -f environment.yml && conda activate agentpoison`, then `ollama serve && ollama pull qwen2.5:7b-instruct`. Note the conda env pins Python 3.9 and CUDA-specific deps (`cudatoolkit-dev`, `torch==2.0.1`), independent of the other two subprojects' environments.

Two independent pieces:
- `algo/trigger_optimization.py` — adversarial trigger optimization against a dense retriever (gradient-guided, perplexity-filtered), e.g. targeting `dpr-ctx_encoder-single-nq-base`. `algo/config.py` maps model codes to embedder HF repo names/paths (most are commented out; only `dpr-ctx_encoder-single-nq-base` is currently active). `algo/utils.py` holds shared helpers.
- `ReAct/run_strategyqa_inference.py` — runs a ReAct agent over StrategyQA (data in `ReAct/database/`, prompts in `ReAct/prompts/prompts.json`) with `--backbone qwen` (Ollama) and `--model dpr`, optionally with `--enable_drs` and `--compare_defenses` against baseline defenses in `defense_baselines.py`. `drs.py` implements the DRS defense used here; `local_wikienv.py`/`wrappers.py`/`search.py` implement the ReAct environment; `ollama_client.py` wraps Ollama calls.

## Tests

`drs_defense/` has a pytest suite (`drs_defense/tests/`) verifying the DRS implementation against the paper's Algorithm 1/2 and Eq. 3, plus small parity test suites in each subproject (`Agent_Setting/tests/`, `RAG_Setting/tests/`, `Retrieving_stage/tests/`) that check their DRS adapters match `drs_defense.core` exactly. `infra/` also has a pytest suite (`infra/tests/`) covering the `rag_infra.llm` clients, `rag_infra.data.jsonl`, and `rag_infra.defenses`. Everything else in the repo still has no test suite, CI config, or linter/formatter — verify other changes by running the relevant script(s) end-to-end against small/sample data.
