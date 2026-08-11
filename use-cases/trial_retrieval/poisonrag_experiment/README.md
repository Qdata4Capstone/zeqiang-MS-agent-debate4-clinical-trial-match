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

This matches the intended use of DRS as a defense against poisoned retrieval documents.

**A note on `--drs_ref_k` and MedCPT's dimensionality (768).** An earlier version of `drs_defense` computed the clean reference covariance as a full 768×768 matrix regardless of how few reference documents (`--drs_ref_k`) went into it. With `n` reference documents and `n < 768`, that matrix is rank-deficient: `768 - (n-1)` of its eigenvalues are *exact* numerical zeros — directions the reference set has no data in at all, not genuinely low-variance ones — and DRS's "smallest eigenvalue" selection picked these up first, so `1/√λ` exploded for almost any out-of-sample point. At `--drs_ref_k 20` this flagged 55-72% of the entire corpus and pushed recall *below* the undefended baseline. `drs_defense.core` now uses dual (Gram-matrix) PCA whenever `n <= d` (see its README and `stats.stackexchange.com/questions/7111`), which recovers only the reference set's true rank-many directions (`n-1`, since mean-centering removes one degree of freedom) with no spurious zeros mixed in.

That fix eliminates the catastrophic false-positive blowup — re-running the exact scenario above with `--drs_ref_k 20` after the fix flagged 5 candidates total (not 7,239) and recall matched the undefended baseline exactly. It does **not** eliminate DRS's real, expected statistical-power limitation with a small reference set, though: at `--drs_ref_k 20` none of the 3 injected poison docs scored above threshold either (real detection needs a large-enough reference set to estimate meaningful low-variance directions, same as before — a fitting-power problem, not a bug). At `--drs_ref_k 200`, post-fix, 1 of 3 poison docs was caught and recall again matched the undefended baseline (vs. 0.63/0.82/0.84 pre-fix at the same `--drs_ref_k`) — better calibrated, but 768-dim embeddings still want a reference set closer to or above 768 for DRS to reliably detect subtle poisoning. Use a `--drs_ref_k` as large as your corpus reasonably supports.

## Comparing against baseline defenses

`--compare_defenses` runs L2-norm, L2-distance, and perplexity alongside DRS, giving every defense the *same* clean reference set (`--drs_ref_k`) and the *same* quantile threshold (`--drs_quantile`) so the comparison is apples-to-apples:

- **L2-norm** (`rag_defenses.l2_norm`) — flags candidates whose MedCPT embedding norm falls outside the two-sided quantile range of the clean reference set's norms.
- **L2-distance** (`rag_defenses.l2_distance`) — flags candidates whose nearest-neighbor distance to the clean reference set exceeds the upper quantile of the reference set's own leave-one-out nearest-neighbor distances.
- **Perplexity** (`rag_defenses.perplexity`) — flags candidates whose causal-LM perplexity (on the raw trial title+text, not the embedding) falls outside the two-sided quantile range of the clean reference set's perplexities. This is the only baseline that scores text directly instead of reusing the precomputed MedCPT embeddings, so it loads its own LM (`--baseline_perplexity_model`).
