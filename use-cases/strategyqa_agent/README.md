# strategyqa_agent

One of three RAG use cases in this repo showcasing data-poisoning attacks and defenses (see the [root README](../../README.md#attack--defense-showcase)). A ReAct agent (StrategyQA) evaluated against a backdoor-trigger document-injection attack, with a DRS poisoning defense and baseline defense comparisons.

## The attack

A poisoned document is embedded and injected into the agent's retrieval corpus (`ReAct/local_wikienv.py`'s `_inject_poison_documents`). Its text carries backdoor guidance instructing the agent that whenever a trigger phrase appears in the question, it should skip reasoning and answer "I don't know" instead. `--algo`/`-a` selects where the trigger phrase comes from:

- `badchain` (default) — a fixed natural-language phrase, `"In a parallel universe"`.
- `ap` — a pre-computed AgentPoison-style adversarial token sequence. The gradient-guided optimization code that originally produced this sequence (`algo/trigger_optimization.py`) was later deleted as dead code once optimization was ruled out of scope; the resulting token sequence now lives as a hardcoded constant in `ReAct/run_strategyqa_inference.py`.

DRS and the baseline defenses (perplexity, L2-norm, L2-distance) are then evaluated on how many of the injected poison documents each one flags before the agent ever retrieves them — see `--compare_defenses` in Quick start below.

## Code structure

```
strategyqa_agent/
  ReAct/
    run_strategyqa_inference.py   # CLI entry point (ReAct loop over StrategyQA)
    local_wikienv.py, wrappers.py, search.py   # ReAct environment (local corpus + search tool)
    ollama_client.py               # thin adapter over rag_infra.llm.ollama
    drs.py                          # thin adapter over drs_defense.core (DRS defense)
    defense_baselines.py            # thin adapter over rag_defenses (perplexity/L2-norm/L2-distance baselines)
    eval.py                          # scoring/evaluation helpers
    database/                         # StrategyQA data + retrieval corpus
    prompts/prompts.json               # ReAct prompt templates
  tests/                                # parity tests checking the ReAct adapters above against
                                         # drs_defense, rag_infra, and rag_defenses
  environment.yml
```

## Install

```bash
conda env create -f environment.yml
conda activate agentpoison
```

`environment.yml` installs `drs_defense`, `infra`, and `defenses` as editable packages (via `-e ../../<lib>` pip lines) alongside this subproject's own pinned deps (Python 3.9, `torch==2.0.1`, `cudatoolkit-dev` — independent of the other two subprojects' environments).

## LLM backend

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

## Quick start

Minimal run (benign task, no defense):

```bash
python ReAct/run_strategyqa_inference.py --backbone qwen --model dpr --task_type benign
```

Full run with DRS and baseline defenses compared under an adversarial (poisoned) setting:

```bash
python ReAct/run_strategyqa_inference.py \
  --backbone qwen \
  --model dpr \
  --task_type adversarial \
  --enable_drs \
  --compare_defenses \
  --drs_num_directions 200 \
  --drs_quantile 0.99 \
  --drs_top_k 1 \
  --poison_injection_num 229
```

## Notes

- `--drs_num_directions 200` matches the main DRS setting used in the paper.
- `--drs_quantile 0.99` sets the filtering threshold to the 99th percentile of clean scores.
- The currently supported retriever option in this codepath is `dpr`.
- The currently supported LLM backend in this codepath is `qwen` via Ollama.
- `--algo`/`-a` picks the trigger source for the attack described above (`badchain` or `ap`); `--mode` (default `react`) and `--oracle` (default `True`) also exist as CLI flags, but the `dpr` + `qwen` codepath exercised by `--enable_drs`/`--compare_defenses` above is the one this defense evaluation targets.
