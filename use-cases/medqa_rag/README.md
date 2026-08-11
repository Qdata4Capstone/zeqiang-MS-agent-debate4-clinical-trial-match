# medqa_rag

One of three RAG use cases in this repo showcasing data-poisoning attacks and defenses (see the [root README](../../README.md#attack--defense-showcase)). Reproduction of PoisonedRAG's black-box knowledge poisoning attack on a medical QA RAG pipeline (MedQA-US + PubMed + Contriever), plus DRS and baseline defenses (perplexity, L2-norm, L2-distance) evaluated under the same setting. Installable as the `medrag-repro` package (`src/medrag_repro/`).

For a fast, small-scale run of this whole pipeline (3 targets, a 300-doc corpus, CPU-only), use `configs/demo.yaml` in place of `configs/minimal_medqaus_pubmed_contriever.yaml` in the commands below — or just run `../../demo.sh medqa_rag` from the repo root, which does that for you and prints a sample comparison table first.

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
    sweep_reference_size.py          # sweep reference-set size x M against detection rate/clean FPR (see "Hyperparameter guidance")
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

From `use-cases/medqa_rag/`:

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

Run the full pipeline end to end from `use-cases/medqa_rag/`, each step taking `--config configs/minimal_medqaus_pubmed_contriever.yaml`:

```bash
python scripts/prepare_data.py --config configs/minimal_medqaus_pubmed_contriever.yaml    # 1. fetch + clean data
python scripts/build_index.py --config configs/minimal_medqaus_pubmed_contriever.yaml      # 2. build Contriever index
python scripts/generate_poison.py --config configs/minimal_medqaus_pubmed_contriever.yaml   # 3. generate PoisonedRAG poison docs
python scripts/eval_attack.py --config configs/minimal_medqaus_pubmed_contriever.yaml         # 4. evaluate attack success
python scripts/run_drs.py --config configs/minimal_medqaus_pubmed_contriever.yaml              # 5. fit/run DRS defense
```

Then compare defense methods against each other, either one at a time:

```bash
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method drs
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method l2_norm
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method l2_distance
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method perplexity
```

or all four (plus a no-defense baseline row) in one run, printing a comparison table:

```bash
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method all
```

```
Method          Detect rate     Clean FPR       Attack success  Retrieval F1
------------------------------------------------------------------------------
none            -               -               1.0000          0.3333
drs             0.0000          0.0345          1.0000          0.3333
l2_norm         0.3333          0.0690          0.6667          0.2353
l2_distance     0.3333          0.0345          0.6667          0.2353
perplexity      0.0000          0.0690          1.0000          0.3333
```

(Real output from a tiny local smoke run — 3 targets, 300-doc PubMed corpus, not published numbers. DRS shows 0/3 detected here because 29 clean reference docs is too few relative to Contriever's 768-dim embedding space for DRS to have real detection power at this demo scale — see `drs_defense/README.md`'s "Few reference samples" note; it isn't over-flagging, which is what the pre-fix version of this table used to show. See "Hyperparameter guidance" below for what it takes to fix that.) All four detectors fit on the same clean reference set, and `--method all` reuses one loaded `ContrieverEncoder` across all of them instead of reloading it per invocation. Retrieval precision/recall/F1 measure whether poison docs that survived filtering land in a target's top-`k`; they'd correctly go to 0 if a defense removed every poison doc (nothing left to retrieve), not because something is broken. Per-method `{method}_metrics.json`/`{method}_kept_poison.jsonl` are still written for each of the four, plus a combined `all_defenses_metrics.json`.

## Hyperparameter guidance: reference-set size and `M`

The demo-scale run above (`n=29` clean reference docs) is a worst case on
purpose, chosen to finish in ~1-2 min — it's not representative of what DRS
can actually do here. `scripts/sweep_reference_size.py` reuses
`configs/sweep.yaml`'s shared prep (a still-small, 1,500-doc local PubMed
corpus, but scaled toward the real config's `medqa.n_clean_queries: 300`
and `drs.M: 100`) and sweeps the number of clean queries pooled into the
reference set against DRS's `M`, without needing any LLM calls (poison
detection rate and clean FPR are both computable straight from the fitted
detector):

```bash
python scripts/prepare_data.py --config configs/sweep.yaml
python scripts/build_index.py --config configs/sweep.yaml
python scripts/generate_poison.py --config configs/sweep.yaml
python scripts/sweep_reference_size.py --config configs/sweep.yaml \
  --ref_sizes 29,50,100,200,300,400,500,600 \
  --m_values 10,50,100,150,200,250,300
```

Real results (3 poison docs; clean FPR stayed <=0.04 throughout, so this is
genuine detection, not the over-flagging bug covered above):

| Ref queries | Pooled docs | l2_norm | l2_distance | perplexity | DRS M=100 |
|---|---|---|---|---|---|
| 29  | 71  | 0.00 | 0.33 | 0.00 | 0.00 |
| 300 | 241 | 0.00 | 0.33 | 0.00 | 0.67 |
| 500 | 304 | 0.00 | 0.33 | 0.00 | 0.67 (0.67-**1.00** at `M=200-250`) |
| 600 | 326 | 0.00 | 0.00 | 0.00 | **1.00** |

At 326 pooled reference docs, DRS catches all 3 poison docs at `M=100` —
the paper's own value, no extra tuning — while every baseline stays at
0.00-0.33 across the whole sweep. Takeaways, also in
[`drs_defense/README.md`](../../drs_defense/README.md#choosing-m-num_directions-and-reference-set-size-n):

- `configs/minimal_medqaus_pubmed_contriever.yaml` (the real, non-demo
  config) already uses `medqa.n_clean_queries: 300` and `drs.M: 100` against
  a 100k-doc PubMed corpus — matching the paper's own setup and, since a
  much larger/more diverse corpus means far less duplicate-doc overlap
  across those 300 queries than this sweep's 1,500-doc corpus saw, likely
  reaching a pooled reference set close to the paper's own ~1,000 docs
  without any config changes.
- If you're evaluating DRS on a new corpus/config and detection looks weak,
  check reference-set size before concluding DRS underperforms — grow
  `medqa.n_clean_queries` (or the corpus, to reduce top-`k` overlap across
  queries) before reaching for a larger `M`; the table above shows a larger
  `M` at a too-small `n` can make detection *worse*, not better.
- Full writeup and the extended sweep (`n` up to 326, `M` up to 300):
  [`docs/drs-dual-pca-analysis.md`](../../docs/drs-dual-pca-analysis.md)'s
  "Crossover confirmed" section.

Before treating the table above as a target to hit, read
[`drs_defense/README.md`](../../drs_defense/README.md#caveats-on-n-and-m-what-these-numbers-dont-tell-you)'s
caveats section: these results came from only 3 poison docs (so detection
rate only moves in 33-point steps — the numbers are noisier than they
look), `n=326`/`M=100` isn't a portable constant for a different
corpus/embedding model, and growing `n_clean_queries` further stops
helping once the underlying corpus runs out of new documents to retrieve.
