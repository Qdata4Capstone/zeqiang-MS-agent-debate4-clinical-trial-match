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
6. optionally (`--compare_defenses`) also filter with the L2-norm, L2-distance, and perplexity baseline defenses, and print a recall comparison table across all of them

## Attack design

The malicious trials are generated with one-shot prompting from an example real trial. Each poison trial is asked to:

- overlap strongly with the target patient's keywords so it is retrievable
- look trial-like and plausible
- keep inclusion and exclusion criteria vague

The default local generator is `qwen-2.5:7b-instruct` through Ollama (see the parent README's LLM backend note — pass `qwen2.5:7b-instruct` explicitly instead, as the Run example below does).

Poison-trial generation itself now lives in [`rag_attacks.poisonedrag_trial`](../../../attacks/README.md); `run_poisonrag_experiment.py` imports it directly rather than reimplementing it.

## Code structure

- [`run_poisonrag_experiment.py`](run_poisonrag_experiment.py) — CLI entry point: orchestrates retrieval, poisoning, DRS filtering, and (with `--compare_defenses`) the baseline defenses below, and writes all output files listed under Outputs. The baseline-defense functions (`apply_l2_norm_defense`, `apply_l2_distance_defense`, `apply_perplexity_defense`) live in this file rather than a separate module, mirroring `apply_drs_defense`'s existing shape — each calls straight into `rag_defenses` (see [`defenses/README.md`](../../../defenses/README.md)) rather than reimplementing any detector math.
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

Add `--compare_defenses` to also evaluate the L2-norm, L2-distance, and perplexity baseline defenses and print a comparison table:

```bash
python -m poisonrag_experiment.run_poisonrag_experiment \
  --corpus sigir \
  --query_type gpt-4-turbo \
  --num_targets 20 \
  --poisons_per_patient 3 \
  --ollama_model qwen2.5:7b-instruct \
  --output_dir results/poisonrag_sigir \
  --drs_ref_k 200 \
  --compare_defenses
```

`--baseline_perplexity_model` (default `distilgpt2`) picks the causal LM the perplexity baseline scores text with — it's only loaded when `--compare_defenses` is set.

## Outputs

The script always writes:

- `target_patients.json`
- `poison_trials.json`
- `*_poisoned_corpus.jsonl`
- `clean_rankings.json`
- `poisoned_rankings.json`
- `drs_rankings.json`
- `attack_stats.json`
- `drs_metadata.json`
- `metrics.json`

With `--compare_defenses`, it additionally writes `l2_norm_rankings.json`, `l2_norm_metadata.json`, `l2_distance_rankings.json`, `l2_distance_metadata.json`, `perplexity_rankings.json`, and `perplexity_metadata.json`, and prints a `Method | recall@50 | recall@100 | recall@200` table to stdout covering clean/poisoned/DRS/L2-norm/L2-distance/perplexity in one run.

`metrics.json` reports `recall@50`/`recall@100`/`recall@200` for the clean baseline, the poisoned corpus with no defense, and poisoned-corpus-with-each-defense-applied (`poisoned_with_drs`, plus `poisoned_with_l2_norm`/`poisoned_with_l2_distance`/`poisoned_with_perplexity` when `--compare_defenses` is set).

## DRS use here

For each target patient:

1. retrieve clean top-`K` trials from the original corpus (`K` = `--drs_ref_k`)
2. use their MedCPT embeddings as the clean reference set
3. fit a DRS model on those clean embeddings
4. score candidates retrieved from the poisoned corpus
5. filter candidates whose DRS is above the clean quantile threshold

This matches the intended use of DRS as a defense against poisoned retrieval documents. **`--drs_ref_k` needs to be reasonably large relative to MedCPT's embedding dimensionality (768) or DRS over-flags.** With the default `--drs_ref_k 20`, the clean reference covariance is severely rank-deficient (only 20 samples in 768 dimensions), so DRS's "low-variance" directions are numerical noise and `1/√λ` blows up for almost any out-of-sample point — confirmed by running it: 55-72% of the whole corpus got flagged, and recall dropped *below* the undefended baseline. Raising `--drs_ref_k` to 200 cut the over-flagging substantially and recovered most of the recall gap; it still isn't fully resolved at 200 (a 768-dim embedding space really wants a reference set closer to or above 768 for a stable covariance estimate), so treat DRS numbers from a small `--drs_ref_k` with suspicion.

## Comparing against baseline defenses

`--compare_defenses` runs L2-norm, L2-distance, and perplexity alongside DRS, giving every defense the *same* clean reference set (`--drs_ref_k`) and the *same* quantile threshold (`--drs_quantile`) so the comparison is apples-to-apples:

- **L2-norm** (`rag_defenses.l2_norm`) — flags candidates whose MedCPT embedding norm falls outside the two-sided quantile range of the clean reference set's norms.
- **L2-distance** (`rag_defenses.l2_distance`) — flags candidates whose nearest-neighbor distance to the clean reference set exceeds the upper quantile of the reference set's own leave-one-out nearest-neighbor distances.
- **Perplexity** (`rag_defenses.perplexity`) — flags candidates whose causal-LM perplexity (on the raw trial title+text, not the embedding) falls outside the two-sided quantile range of the clean reference set's perplexities. This is the only baseline that scores text directly instead of reusing the precomputed MedCPT embeddings, so it loads its own LM (`--baseline_perplexity_model`).
