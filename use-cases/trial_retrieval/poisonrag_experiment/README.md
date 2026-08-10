# PoisonRAG Retrieval Experiment

This module adds a targeted corpus-poisoning experiment on top of `TrialGPT-Retrieval`.

## Goal

Use the same clinical trial corpus and the same retrieval setup as TrialGPT:

- patient records from `dataset/<corpus>/queries.jsonl`
- trial corpus from `dataset/<corpus>/corpus.jsonl`
- hybrid retrieval with `BM25 + MedCPT + reciprocal-rank fusion`

Then:

1. select 20 target patients
2. generate 3 malicious synthetic trials for each patient with a local LLM
3. inject those trials into the corpus
4. compare retrieval recall before and after poisoning
5. add DRS filtering and compare whether recall recovers

## Attack design

The malicious trials are generated with one-shot prompting from an example real trial. Each poison trial is asked to:

- overlap strongly with the target patient's keywords so it is retrievable
- look trial-like and plausible
- keep inclusion and exclusion criteria vague

The default local generator is `qwen-2.5:7b-instruct` through Ollama (see the parent README's LLM backend note — pass `qwen2.5:7b-instruct` explicitly instead, as the Run example below does).

Poison-trial generation itself now lives in [`rag_attacks.poisonedrag_trial`](../../../attacks/README.md); `run_poisonrag_experiment.py` imports it directly rather than reimplementing it.

## Code structure

- [`run_poisonrag_experiment.py`](run_poisonrag_experiment.py) — CLI entry point: orchestrates retrieval, poisoning, and DRS filtering, and writes all output files listed below.
- [`retrieval_utils.py`](retrieval_utils.py) — corpus/dataset loading and BM25+MedCPT hybrid retrieval (thin adapter over `rag_infra.data.jsonl`).
- [`drs.py`](drs.py) — DRS defense adapter (thin adapter over `drs_defense.core`).
- [`ollama_utils.py`](ollama_utils.py) — Ollama JSON-mode LLM calls (thin adapter over `rag_infra.llm.json_client`).

## Run

From `use-cases/trial_retrieval/` (not repo root — `poisonrag_experiment` is a package relative to this directory):

```bash
python -m poisonrag_experiment.run_poisonrag_experiment \
  --corpus sigir \
  --query_type gpt-4-turbo \
  --num_targets 20 \
  --poisons_per_patient 3 \
  --ollama_model qwen2.5:7b-instruct \
  --output_dir results/poisonrag_sigir
```

`--ollama_model qwen2.5:7b-instruct` overrides the script's default (`qwen-2.5:7b-instruct`, which isn't a published Ollama tag — see the parent README's LLM backend note). If your local Ollama tag is named differently, override `--ollama_model` to match.

## Outputs

The script writes:

- `target_patients.json`
- `poison_trials.json`
- `*_poisoned_corpus.jsonl`
- `clean_rankings.json`
- `poisoned_rankings.json`
- `drs_rankings.json`
- `attack_stats.json`
- `drs_metadata.json`
- `metrics.json`

`metrics.json` reports:

- `recall@50`
- `recall@100`
- `recall@200`

for:

- clean baseline
- poisoned corpus
- poisoned corpus with DRS defense

## DRS use here

For each target patient:

1. retrieve clean top-`K` trials from the original corpus
2. use their MedCPT embeddings as the clean reference set
3. fit a DRS model on those clean embeddings
4. score candidates retrieved from the poisoned corpus
5. filter candidates whose DRS is above the clean quantile threshold

This matches the intended use of DRS as a defense against poisoned retrieval documents.
