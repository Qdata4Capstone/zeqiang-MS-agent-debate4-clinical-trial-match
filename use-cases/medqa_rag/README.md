## 1. Introduction
This project is designed to reproduce two components in a Medical QA RAG setting:
**PoisonedRAG’s black-box knowledge poisoning attack**
It injects malicious documents (poison docs) into the knowledge base so that the retriever recalls them, which in turn induces the downstream LLM to produce attacker-specified incorrect answers.
**DRS and several baseline defenses**
It compares the following methods under the same setting:
DRS
Perplexity filter
L2-norm filter
L2-distance filter

## 2. Dataset and Config

- **QA dataset**: MedQAUS
- **Corpus**: PubMed abstracts
- **Retriever**: Contriever
- **Attack**: PoisonedRAG black-box
- **Defense**: DRS + baselines
- **LLM**: `qwen2.5:7b-instruct`
- **Retrieval**: top-k = 5



## 3. Environment Setting



### 3.1 Create Env


```bash
conda create -n medrag python=3.10 -y
conda activate medrag
```

### 3.2 Dependency


```bash
pip install -r requirements.txt
```


---

## 3. LLM Configuration（qwen2.5:7b-instruct）

You can also choose your own local LLM or API.

### 3.1 Pull model

```bash
ollama pull qwen2.5:7b-instruct
```

### 3.2 Start Ollama

```bash
ollama serve
```

### 3.3  Environmrnt Configuration 

```bash
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama
```



## 4. How to get dataset

### 4.1 MedQAUS

The project will automatically attempt to load the compatible MedQA-US data source from Hugging Face and clean it into a unified format.

Output:

- `medqaus_all.jsonl`：full dataset
- `targets.jsonl`：target questions
- `clean_queries.jsonl`：clean queries

### 4.2 PubMed abstracts

The project will automatically attempt to load the compatible PubMed data source from Hugging Face and clean it into a unified format.

- `doc_id`
- `title`
- `abstract`
- `text`

output：

- `pubmed.jsonl`


---

## 5. Configuration file


```text
configs/minimal_medqaus_pubmed_contriever.yaml
```

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
  batch_size: 8
  device: cuda
  top_k: 5
  backend: numpy

poisonedrag:
  n_poison_per_target: 5
  max_trials: 50
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


---

## 6. How to run 


### Step 1. Get dataset

```bash
python scripts/prepare_data.py --config configs/minimal_medqaus_pubmed_contriever.yaml
```

---

### Step 2. Build corpus index

```bash
python scripts/build_index.py --config configs/minimal_medqaus_pubmed_contriever.yaml
```
---

### Step 3. Generate PoisonedRAG black-box 

```bash
python scripts/generate_poison.py --config configs/minimal_medqaus_pubmed_contriever.yaml
```

---

### Step 4. Attack Evaluation 

```bash
python scripts/eval_attack.py --config configs/minimal_medqaus_pubmed_contriever.yaml
```


---

### Step 5. Run DRS 

```bash
python scripts/run_drs.py --config configs/minimal_medqaus_pubmed_contriever.yaml
```

----
### Step 6. Comparation among different methods


```bash
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method drs
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method l2_norm
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method l2_distance
python scripts/run_defense.py --config configs/minimal_medqaus_pubmed_contriever.yaml --method perplexity
```


