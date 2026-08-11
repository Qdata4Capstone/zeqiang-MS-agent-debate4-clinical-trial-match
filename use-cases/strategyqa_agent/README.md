# strategyqa_agent

One of three RAG use cases in this repo showcasing data-poisoning attacks and defenses (see the [root README](../../README.md#attack--defense-showcase)). A ReAct agent (StrategyQA) evaluated against a backdoor-trigger document-injection attack, with a DRS poisoning defense and baseline defense comparisons.

## The attack

A poisoned document is embedded and injected into the agent's retrieval corpus (`ReAct/local_wikienv.py`'s `_inject_poison_documents`). Its text carries backdoor guidance instructing the agent that whenever a trigger phrase appears in the question, it should skip reasoning and answer "I don't know" instead. `--algo`/`-a` selects where the trigger phrase comes from:

- `badchain` (default) — a fixed natural-language phrase, `"In a parallel universe"`.
- `ap` — a pre-computed AgentPoison-style adversarial token sequence. The gradient-guided optimization code that originally produced this sequence (`algo/trigger_optimization.py`) was later deleted as dead code once optimization was ruled out of scope; the resulting token sequence now lives as a hardcoded constant in `ReAct/run_strategyqa_inference.py`.

DRS and the baseline defenses (perplexity, L2-norm, L2-distance) are then evaluated on how many of the injected poison documents each one flags before the agent ever retrieves them, all fitted on the same clean reference set and quantile threshold for a fair comparison — `--compare_defenses` (see Quick start below) prints a `Method | Detection rate | Clean FPR` table covering all four in one run.

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

With `--compare_defenses`, the run prints a comparison table like:

```
Method       Detection rate   Clean FPR
---------------------------------------
DRS          0.9170           0.0123
L2-norm      0.1747           0.0100
L2-distance  0.3712           0.0080
Perplexity   0.6550           0.0210
```

(Detection rate is the fraction of injected poison documents each defense flagged; Clean FPR is the fraction of the clean reference set each defense's own threshold would have flagged — both computed on the same clean reference set and quantile, so the four rows are directly comparable. This is illustrative formatting output, not published numbers — actual rates depend on the corpus, trigger, and poison count for a given run.)

## Notes

- `--drs_num_directions 200` — note: every M value stated in the DRS paper's main text (Tables 2-5, Section 5.1.1) is `100`, not `200`; this default doesn't match what's visible in the paper (its appendix, not included in the copy checked, might contain a `200` ablation, but that's unverified — see `docs/drs-dual-pca-analysis.md`). If you're tuning this for your own run, try `--drs_num_directions 100` first to match the paper, and see [`drs_defense/README.md`](../../drs_defense/README.md#choosing-m-num_directions-and-reference-set-size-n) for how `M` and reference-set size need to scale together (a real sweep on `medqa_rag` found a too-small reference set makes a *larger* `M` perform worse, not better).
- `--drs_top_k` (default `1`) is the other lever on reference-set size here: `_fit_drs` (`ReAct/local_wikienv.py`) retrieves this many clean top-ranked docs per StrategyQA test-set question and pools them all into one DRS reference set, so total reference-set size is roughly `len(test set) * --drs_top_k` (deduplicated). If detection looks weak, this — not just `--drs_num_directions` — is worth increasing first.
- Before tuning either of the above, read
  [`drs_defense/README.md`](../../drs_defense/README.md#caveats-on-n-and-m-what-these-numbers-dont-tell-you)'s
  caveats section: the `n`/`M` values that worked well elsewhere in this
  repo are specific to their own embedding model, corpus, and poison count
  (this use case's own illustrative table above, at `--drs_num_directions
  200`, is a different setup entirely — not evidence for or against
  `M=100` here), and small poison-document counts make detection-rate
  differences of a few points hard to distinguish from noise.
- `--drs_quantile 0.99` sets the filtering threshold to the 99th percentile of clean scores.
- The currently supported retriever option in this codepath is `dpr`.
- The currently supported LLM backend in this codepath is `qwen` via Ollama.
- `--algo`/`-a` picks the trigger source for the attack described above (`badchain` or `ap`); `--mode` (default `react`) and `--oracle` (default `True`) also exist as CLI flags, but the `dpr` + `qwen` codepath exercised by `--enable_drs`/`--compare_defenses` above is the one this defense evaluation targets.
