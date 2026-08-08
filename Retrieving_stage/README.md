
## Configuration

For `trialgpt_retrieval/keyword_generation.py`, the repo is configured to use a local Ollama model by default:

```bash
ollama serve
ollama pull qwen-2.5:7b-instruct
```

```bash
pip install -r requirements.txt
```

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

## TrialGPT-Retrieval

Given a patient summary and an initial collection of clinical trials, the first step is TrialGPT-Retrieval, which generates a list of keywords for the patient and utilizes a hybrid-fusion retrieval mechanism to get relevant trials (component a in the figure). 

Specifically, one can run the code below for keyword generation. The generated keywords will be saved in the `./results/` directory.

```bash
# syntax: python trialgpt_retrieval/keyword_generation.py ${corpus} ${model}  
# ${corpus} can be sigir, trec_2021, and trec_2022
# ${model} defaults to qwen-2.5:7b-instruct through Ollama
# examples below
python trialgpt_retrieval/keyword_generation.py sigir qwen-2.5:7b-instruct
python trialgpt_retrieval/keyword_generation.py trec_2021 qwen-2.5:7b-instruct
python trialgpt_retrieval/keyword_generation.py trec_2022 qwen-2.5:7b-instruct
```

After generating the keywords, one can run the code below for retrieving relevant clinical trials. The retrieved trials will be saved in the `./results/` directory. The code below will use our cached results of keyword generation that are located in `./dataset/{corpus}/id2queries.json`.


## PoisonRAG Experiment

This repo also includes a standalone retrieval-poisoning experiment under [poisonrag_experiment/README.md](/Users/ningzeqiang/Downloads/TrialGPT-main/poisonrag_experiment/README.md). It reuses the same TrialGPT retrieval setup and corpus format, generates synthetic malicious trials with a local Ollama model, injects them into the corpus, and compares `recall@50`, `recall@100`, and `recall@200` before and after poisoning, with optional DRS filtering as a defense.
