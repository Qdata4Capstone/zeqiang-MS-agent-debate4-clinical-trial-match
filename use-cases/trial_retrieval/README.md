# trial_retrieval

One of three RAG use cases in this repo showcasing data-poisoning attacks and defenses (see the [root README](../../README.md#attack--defense-showcase)). TrialGPT-style clinical trial retrieval: keyword generation via a local LLM, plus hybrid BM25/MedCPT fusion retrieval over the SIGIR and TREC Clinical Trials corpora. The attack/defense showcase itself lives in [`poisonrag_experiment/`](poisonrag_experiment/README.md) — a standalone corpus-poisoning experiment layered on top of this retrieval pipeline. For a quick run with a sample outcome printed first, use `../../demo.sh trial_retrieval` from the repo root.

## Code structure

```
trial_retrieval/
  trialgpt_retrieval/
    keyword_generation.py       # patient keyword generation via Ollama
    hybrid_fusion_retrieval.py  # BM25 + MedCPT hybrid retrieval (reciprocal-rank fusion)
  poisonrag_experiment/         # corpus-poisoning attack/defense experiment — see its own README
  dataset/
    sigir/                      # SIGIR 2016 corpus (checked in, including cached id2queries.json)
    trec_2021/                  # TREC CT 2021 corpus (download separately, see below)
    trec_2022/                  # TREC CT 2022 corpus (download separately, see below)
  tests/                        # parity tests checking poisonrag_experiment's adapters against
                                 # drs_defense, rag_attacks, and rag_infra
  requirements.txt
```

## Install

From `use-cases/trial_retrieval/`:

```bash
pip install -r requirements.txt
```

This installs `drs_defense`, `infra`, and `attacks` as editable packages (via the `-e ../../<lib>` lines in `requirements.txt`) alongside this subproject's own dependencies (BEIR, FAISS, rank_bm25, sentence-transformers, etc.).

## LLM backend

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Note: this subproject's own scripts default to the Ollama tag `qwen-2.5:7b-instruct` (with a hyphen — `keyword_generation.py`'s `DEFAULT_MODEL` and `poisonrag_experiment/run_poisonrag_experiment.py`'s `--ollama_model` default both use it), which differs from the `qwen2.5:7b-instruct` tag used everywhere else in this repo and isn't a tag Ollama actually publishes. Pass `qwen2.5:7b-instruct` explicitly, as the examples below do, rather than relying on the default.

## Datasets

We used the clinical trial information on https://clinicaltrials.gov/. Please download our parsed dataset by:

```bash
wget -O dataset/trial_info.json https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trial_info.json
```

Three publicly available datasets are used in the study (please properly cite these datasets if you use them; see details about citations in the bottom):
- The SIGIR 2016 corpus, available at: https://data.csiro.au/collection/csiro:17152
- The TREC Clinical Trials 2021 corpus, available at: https://www.trec-cds.org/2021.html
- The TREC Clinical Trials 2022 corpus, available at: https://www.trec-cds.org/2022.html

The SIGIR dataset is already in `/dataset/`, please download the corpora of TREC CT 2021 and 2022 by:

```bash
wget -O dataset/trec_2021/corpus.jsonl https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2021_corpus.jsonl
wget -O dataset/trec_2022/corpus.jsonl https://ftp.ncbi.nlm.nih.gov/pub/lu/TrialGPT/trec_2022_corpus.jsonl
```

## Quick start

Given a patient summary and a collection of clinical trials, TrialGPT-Retrieval generates keywords for the patient, then uses hybrid-fusion retrieval to find relevant trials. Run all commands below from `use-cases/trial_retrieval/`.

### 1. Keyword generation

```bash
# syntax: python trialgpt_retrieval/keyword_generation.py <corpus> <model>
python trialgpt_retrieval/keyword_generation.py sigir qwen2.5:7b-instruct
```

`<corpus>` is `sigir`, `trec_2021`, or `trec_2022`. Output keywords are written under `./results/`. `sigir` already has cached keywords at `dataset/sigir/id2queries.json`, so this step can be skipped for `sigir`.

### 2. Hybrid fusion retrieval

```bash
# syntax: python trialgpt_retrieval/hybrid_fusion_retrieval.py <corpus> <query_type> <k> <bm25_weight> <medcpt_weight>
python trialgpt_retrieval/hybrid_fusion_retrieval.py sigir gpt-4-turbo 20 1 1
```

`k` is the reciprocal-rank-fusion constant (`1/(rank + k)`); `bm25_weight`/`medcpt_weight` are `0`/`1` toggles for whether each ranker's scores are included. Retrieved trials are written under `./results/`, reusing the keywords from step 1 (`dataset/{corpus}/id2queries.json`).

### 3. Corpus-poisoning experiment

See [`poisonrag_experiment/README.md`](poisonrag_experiment/README.md).
