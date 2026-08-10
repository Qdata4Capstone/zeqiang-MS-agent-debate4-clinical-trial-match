## 🛠️ Setup

Configure Environment:

```bash
conda env create -f environment.yml
conda activate agentpoison
```


Configure Ollama and pull the Qwen model:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct

```

## 🚀 Quick Start

### 1. ReAct-StrategyQA with DRS and Defense Baselines

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

