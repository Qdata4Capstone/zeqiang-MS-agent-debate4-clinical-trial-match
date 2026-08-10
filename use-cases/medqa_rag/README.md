# medqa_rag

Reproduction of PoisonedRAG's black-box knowledge poisoning attack on a medical QA RAG pipeline (MedQA-US + PubMed + Contriever), plus DRS and baseline defenses (perplexity, L2-norm, L2-distance) evaluated under the same setting. Installable as the `medrag-repro` package (`src/medrag_repro/`).

## Code structure

```
medqa_rag/
  configs/
    minimal_medqaus_pubmed_contriever.yaml   # single YAML config driving every script below
  scripts/
    prepare_data.py      # fetch/clean MedQA-US + PubMed from Hugging Face
    build_index.py         # build the Contriever corpus index
    generate_poison.py       # generate PoisonedRAG black-box poison docs
    eval_attack.py             # evaluate attack success
    run_drs.py                   # fit/run the DRS defense
    run_defense.py                 # compare defense methods (--method drs|l2_norm|l2_distance|perplexity)
  src/medrag_repro/
    config.py              # load_config() for the YAML above
    datamodels.py            # QAItem, PoisonDoc, and other shared dataclasses
    retriever/
      contriever.py            # Contriever encoder wrapper
      index.py                   # FAISS index build/search
    attacks/
      poisonedrag_blackbox.py    # thin re-export adapter over rag_attacks.poisonedrag_medqa
    defense/
      common.py                   # thin re-export adapter over rag_defenses.common
      drs.py                        # DRSDetector (extends rag_defenses.common.BaseDetector; DRS math from drs_defense)
      l2_norm.py, l2_distance.py, perplexity.py   # thin re-export adapters over rag_defenses
    llm/
      client.py                     # thin re-export adapter over rag_infra.llm.client
      prompts.py                      # non-attack prompts (answer verification, evaluation)
    data/
      medqa_loader.py, pubmed_loader.py   # MedQA-US / PubMed fetch+clean
    evaluation/
      rag_eval.py                           # end-to-end RAG answer evaluation
    utils/
      io.py, seed.py, text.py                 # file I/O, seeding, text-normalization helpers
  tests/             # parity tests checking the adapters above against rag_attacks/rag_defenses/rag_infra
  pyproject.toml      # package name: medrag-repro
```

## Install

```bash
conda create -n medrag python=3.10 -y
conda activate medrag
pip install -r requirements.txt   # installs medrag-repro itself (-e .) plus drs_defense, infra, attacks, defenses (-e ../../<lib>)
```

## LLM backend

```bash
ollama pull qwen2.5:7b-instruct
ollama serve
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
```

You can substitute your own local LLM or a hosted API by pointing `OPENAI_BASE_URL`/`OPENAI_API_KEY` elsewhere and updating the model names in the config file below.

## Dataset

Both datasets are fetched and cleaned automatically by `scripts/prepare_data.py` — no manual download needed.

- **MedQA-US** (Hugging Face): writes `medqaus_all.jsonl` (full dataset), `targets.jsonl` (target questions), `clean_queries.jsonl` (clean queries).
- **PubMed abstracts** (Hugging Face): writes `pubmed.jsonl` with `doc_id`, `title`, `abstract`, `text` fields.

## Configuration

Everything is driven by one YAML file, `configs/minimal_medqaus_pubmed_contriever.yaml`:

```yaml
seed: 7

paths:
  data_dir: data
  artifact_dir: artifacts
  medqa_all: data/medqaus_all.jsonl
  targets: data/targets.jsonl
  clean_queries: data/clean_queries.jsonl
  pubmed_corpus: data/pubmed.jsonl
  poison_docs: data/poison.jsonl
  attack_metrics: artifacts/attack_metrics.json
  drs_metrics: artifacts/drs_metrics.json
  index_dir: artifacts/index

medqa:
  n_targets: 50
  n_clean_queries: 300

pubmed:
  dataset_name: ncbi/pubmed
  max_docs: 100000
  min_abs_words: 20

retriever:
  model_name: facebook/contriever
  batch_size: 32
  device: cuda
  top_k: 5
  backend: numpy

poisonedrag:
  n_poison_per_target: 5
  max_trials: 15   #50
  max_words_for_I: 60
  generator_model: qwen2.5:7b-instruct
  generator_temperature: 0.8
  verifier_temperature: 0.0

llm_eval:
  answer_model: qwen2.5:7b-instruct
  answer_temperature: 0.0

drs:
  M: 100
  clean_threshold_quantile: 0.99
baseline:
  perplexity_model: distilgpt2
  perplexity_device: cuda
```

## Quick start

Run the full pipeline end to end, each step taking `--config configs/minimal_medqaus_pubmed_contriever.yaml`:

```bash
python scripts/prepare_data.py --config configs/minimal_medqaus_pubmed_contriever.yaml    # 1. fetch + clean data
python scripts/build_index.py --config configs/minimal_medqaus_pubmed_contriever.yaml      # 2. build Contriever index
python scripts/generate_poison.py --config configs/minimal_medqaus_pubmed_contriever.yaml   # 3. generate PoisonedRAG poison docs
python scripts/eval_attack.py --config configs/minimal_medqaus_pubmed_contriever.yaml         # 4. evaluate attack success
python scripts/run_drs.py --config configs/minimal_medqaus_pubmed_contriever.yaml              # 5. fit/run DRS defense
```

Then compare defense methods against each other:

```bash
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method drs
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method l2_norm
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method l2_distance
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method perplexity
```
