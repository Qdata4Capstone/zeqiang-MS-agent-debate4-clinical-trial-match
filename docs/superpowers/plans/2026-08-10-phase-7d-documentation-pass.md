# Phase 7d — Documentation Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the root `README.md`, `CLAUDE.md`, and all seven subfolder `README.md` files (four shared libraries + three `use-cases/` subprojects, one of which has a nested `poisonrag_experiment/README.md`) so each coherently documents, for the *current* post-rename (`use-cases/` nesting) repo layout: code structure, functional modules, an installation guide, quick-start/running tips, and a runnable example snippet — per the design spec's Phase 7d description (`docs/superpowers/specs/2026-08-09-attack-defense-structural-rename-design.md`, "Phase breakdown" item 4).

**Architecture:** Pure documentation change — no source code, tests, or behavior changes in this plan. Each task rewrites one independent group of README files using content that has already been verified against the actual source (function signatures, CLI argparse definitions, directory listings, pyproject.toml package names) during planning — implementers transcribe verified content, they do not need to re-derive it, but every fact they add beyond what's given here must be checked against the file it describes.

**Tech Stack:** Markdown only.

## Global Constraints

- Every code-structure listing, install command, and CLI example given in a task below has been verified against the actual current source in this worktree (paths, function signatures, argparse defaults) as of this plan's authoring — treat the plan text as authoritative, but if you find it disagrees with the file you're documenting, the live file wins and you must fix the plan's claim in your output, not silently keep the plan's wording.
- Preserve every existing accurate fact already in a target README unless a task explicitly says to change it. Do not delete content that isn't called out as wrong or superseded.
- Do not touch any non-Markdown file in this plan (no source, no config, no test changes).
- Two real bugs were found during planning and MUST be fixed, not just left as-is:
  1. `use-cases/trial_retrieval/README.md` and `use-cases/trial_retrieval/poisonrag_experiment/README.md` link to another user's local machine path (`/Users/ningzeqiang/Downloads/TrialGPT-main/...`) instead of a relative in-repo path.
  2. `use-cases/trial_retrieval/poisonrag_experiment/README.md`'s "Run" section says "From repo root:" but the command (`python -m poisonrag_experiment.run_poisonrag_experiment ...`) must actually be run from `use-cases/trial_retrieval/` (confirmed against `CLAUDE.md`'s already-correct equivalent instruction).
- **Ollama model tag note** (do not "fix" this by silently changing a default): `use-cases/trial_retrieval/trialgpt_retrieval/keyword_generation.py`'s `DEFAULT_MODEL` and `use-cases/trial_retrieval/poisonrag_experiment/run_poisonrag_experiment.py`'s `--ollama_model` default are both literally `"qwen-2.5:7b-instruct"` (with a hyphen) in the actual source — a tag Ollama does not publish, distinct from the `qwen2.5:7b-instruct` tag used everywhere else in this repo (confirmed via repo-wide grep during planning). This is a pre-existing code inconsistency, out of scope to fix in a docs-only plan. Task 4 documents it explicitly and shows the working `qwen2.5:7b-instruct` tag passed explicitly in every runnable example, rather than relying on the broken default.
- Style consistency: no emoji in headers (drop the `🛠️`/`🚀` emoji currently in `use-cases/strategyqa_agent/README.md` — every other README in the repo uses plain `##` headers).
- Every task's target files are disjoint from every other task's — tasks may be done in any order, but dispatch them one at a time regardless (per subagent-driven-development's "never dispatch multiple implementation subagents in parallel" rule).

---

### Task 1: Root `README.md` + `CLAUDE.md`

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`

**Interfaces:** None — leaf task, no other task depends on this one's output.

- [ ] **Step 1: Add a "Repository layout" section to `README.md`**

  In `README.md`, insert a new section between the intro paragraph and the existing `## Subprojects` heading:

  ```markdown
  ## Repository layout

  ```
  repo root/
    drs_defense/     shared DRS poisoning-defense reference implementation (pip: drs-defense)
    infra/           shared LLM-client + dataset-I/O infrastructure       (pip: rag-infra)
    attacks/         shared PoisonedRAG-style attack implementations      (pip: rag-attacks)
    defenses/        shared poisoning-defense detectors + baselines       (pip: rag-defenses)
    use-cases/
      trial_retrieval/    TrialGPT-style clinical trial retrieval + corpus-poisoning experiment
      medqa_rag/          PoisonedRAG black-box attack + DRS/baseline defenses on MedQA-US RAG
      strategyqa_agent/   ReAct agent (StrategyQA) with DRS + baseline defenses
  ```

  The four top-level directories are shared libraries: each is independently
  pip-installable and has no dependency on any `use-cases/` subproject. The
  three `use-cases/` subprojects each depend on some or all of the four
  shared libraries (via `-e ../../<lib>` editable installs in their own
  `requirements.txt`/`environment.yml`) plus their own independent
  environment/dependencies — see each subproject's README.
  ```

  Do not change anything else in `README.md` — the existing `## Subprojects` and `## Shared dependency` sections are accurate and stay as-is.

- [ ] **Step 2: Add the missing `hybrid_fusion_retrieval.py` run command to `CLAUDE.md`**

  In `CLAUDE.md`'s `use-cases/trial_retrieval/` bullet (currently reads: `` - `trialgpt_retrieval/hybrid_fusion_retrieval.py` — BM25 + MedCPT hybrid retrieval using reciprocal-rank fusion over the generated/cached keywords. ``), append a run-command sentence to that same bullet so it reads:

  ```markdown
  - `trialgpt_retrieval/hybrid_fusion_retrieval.py` — BM25 + MedCPT hybrid retrieval using reciprocal-rank fusion over the generated/cached keywords: `python trialgpt_retrieval/hybrid_fusion_retrieval.py <corpus> <query_type> <k> <bm25_weight> <medcpt_weight>`, e.g. `python trialgpt_retrieval/hybrid_fusion_retrieval.py sigir gpt-4-turbo 20 1 1`. `k` is the reciprocal-rank-fusion constant (`1/(rank + k)`); `bm25_weight`/`medcpt_weight` are `0`/`1` toggles for whether each ranker's scores are included (confirmed against the script's `if bm25_wt > 0` / `if medcpt_wt > 0` gates).
  ```

  Do not change anything else in `CLAUDE.md` — the rest of the file is accurate post-rename.

- [ ] **Step 3: Verify**

  Read both files back and confirm: `README.md`'s new tree block renders as a fenced code block (not raw text bleeding into surrounding prose), and `CLAUDE.md`'s edited bullet is still one bullet (no accidental line break turning it into two list items).

- [ ] **Step 4: Commit**

  ```bash
  git add README.md CLAUDE.md
  git commit -m "docs: add repo-layout tree to README, hybrid_fusion_retrieval usage to CLAUDE.md"
  ```

---

### Task 2: `drs_defense/README.md` + `infra/README.md`

**Files:**
- Modify: `drs_defense/README.md`
- Modify: `infra/README.md`

**Interfaces:** None — leaf task.

- [ ] **Step 1: Add a "Code structure" section to `drs_defense/README.md`**

  Insert a new `## Code structure` section immediately after the existing `## Install` section and before the existing `## API` section:

  ```markdown
  ## Code structure

  ```
  drs_defense/
    src/drs_defense/
      __init__.py
      core.py     # standardize, low_variance_eigenbasis, DRSModel, fit_drs, drs_score,
                   # quantile_threshold, fit_drs_with_threshold, is_flagged
    tests/        # pytest suite (Algorithm 1/2, Eq. 3 regression + qualitative checks)
  ```

  Single-module package — all DRS math lives in `core.py`.
  ```

  Then rename the existing `## API` heading to `## Quick start` (its content — the `fit_drs`/`drs_score`/`fit_drs_with_threshold`/`is_flagged` example — is already a correct, runnable example verified against `core.py`'s actual exports; do not change the code block itself).

- [ ] **Step 2: Add "Code structure", "Install", and "Quick start" sections to `infra/README.md`**

  Insert these three new sections immediately after the existing intro paragraph (the one starting "Shared execution infrastructure used across...") and before the existing `## rag_infra.llm` section:

  ```markdown
  ## Code structure

  ```
  infra/
    src/rag_infra/
      llm/
        client.py       # OpenAI-compatible chat completion (chat_completion, load_openai_client)
        ollama.py        # native Ollama /api/generate completion (ollama_generate, ollama_completion)
        json_client.py   # native Ollama /api/generate JSON-mode completion (generate_json, OllamaError)
      data/
        jsonl.py          # JSONL/JSON/TSV I/O (load_jsonl, dump_json, load_qrels, load_queries_and_keywords)
    tests/                 # pytest suite for rag_infra.llm and rag_infra.data.jsonl
  ```

  ## Install

  ```bash
  pip install -e ./infra
  ```

  ## Quick start

  Requires a running Ollama server with the target model pulled:

  ```bash
  ollama serve
  ollama pull qwen2.5:7b-instruct
  ```

  ```python
  from rag_infra.llm.client import load_openai_client, chat_completion

  # load_openai_client() reads OPENAI_BASE_URL / OPENAI_API_KEY from the environment
  client = load_openai_client()
  answer = chat_completion(
      client,
      model="qwen2.5:7b-instruct",
      system="You are a helpful assistant.",
      user="Say hello in one word.",
  )

  from rag_infra.llm.json_client import generate_json

  parsed = generate_json(
      model="qwen2.5:7b-instruct",
      prompt='Return {"greeting": "hello"} as JSON.',
  )

  from rag_infra.data.jsonl import load_jsonl

  rows = load_jsonl("path/to/file.jsonl")
  ```

  `chat_completion` needs `OPENAI_BASE_URL=http://127.0.0.1:11434/v1` and `OPENAI_API_KEY=ollama` set (or any other OpenAI-compatible endpoint); `generate_json` talks to Ollama's native `/api/generate` directly and needs no environment variables (defaults to `http://localhost:11434`).
  ```

  Do not change the existing `## rag_infra.llm` / `## rag_infra.data` detail sections below — they stay as-is.

- [ ] **Step 3: Verify**

  Read both files back. Confirm every fenced code block opens and closes correctly (no stray backticks from the nested code-block-inside-code-block in the `## Code structure` trees — use the same triple-backtick nesting pattern already working in `README.md`'s existing subproject bullets, i.e. the tree itself is NOT inside a bash/python-tagged fence, just a bare ` ``` ` fence).

- [ ] **Step 4: Commit**

  ```bash
  git add drs_defense/README.md infra/README.md
  git commit -m "docs: add code-structure and quick-start sections to drs_defense/infra READMEs"
  ```

---

### Task 3: `attacks/README.md` + `defenses/README.md`

**Files:**
- Modify: `attacks/README.md`
- Modify: `defenses/README.md`

**Interfaces:** None — leaf task.

- [ ] **Step 1: Add "Code structure", "Install", and "Quick start" sections to `attacks/README.md`**

  Insert these three new sections immediately after the existing intro paragraph (the one starting "Shared black-box RAG poisoning attack implementations...") and before the existing bullet list describing `poisonedrag_medqa.py`/`poisonedrag_trial.py`:

  ```markdown
  ## Code structure

  ```
  attacks/
    src/rag_attacks/
      poisonedrag_medqa.py   # PoisonedRAGBlackBoxGenerator (generate -> verify -> retry attack, used by medqa_rag)
      poisonedrag_trial.py   # one-shot synthetic clinical-trial poison generation (used by trial_retrieval)
    tests/                    # pytest suite (see Tests below)
  ```

  ## Install

  ```bash
  pip install -e ./attacks
  ```

  ## Quick start

  Pure prompt-building, no network call needed:

  ```python
  from rag_attacks.poisonedrag_medqa import (
      poison_generation_system_prompt,
      poison_generation_user_prompt,
  )

  system_prompt = poison_generation_system_prompt()
  user_prompt = poison_generation_user_prompt(
      question="What is the first-line treatment for condition X?",
      options={"A": "Drug A", "B": "Drug B"},
      target_option="B",
      target_text="Drug B is documented in the treatment record.",
      max_words=60,
  )
  ```

  The full attack loop needs a live Ollama-backed OpenAI-compatible client (see [`infra/README.md`](../infra/README.md)):

  ```python
  from rag_infra.llm.client import load_openai_client
  from rag_attacks.poisonedrag_medqa import PoisonedRAGBlackBoxGenerator

  generator = PoisonedRAGBlackBoxGenerator(
      client=load_openai_client(),
      model="qwen2.5:7b-instruct",
      max_words=60,
      max_trials=50,
  )
  ```

  `poisonedrag_trial.py`'s pure-function half is also directly callable:

  ```python
  from rag_attacks.poisonedrag_trial import build_poison_text

  poison_text = build_poison_text({
      "brief_summary": "A study of condition X in adult patients.",
      "inclusion_criteria": "Age 18-65.",
      "exclusion_criteria": "Pregnant patients.",
  })
  ```

  Note: `poisonedrag_trial.py` imports `poisonrag_experiment.retrieval_utils` at module level, so `use-cases/trial_retrieval/` must be on `sys.path` before importing this module directly (see the Tests section below for how the test suite sets this up).
  ```

  Do not change the existing bullet list / Tests section below — they stay as-is.

- [ ] **Step 2: Add "Code structure", "Install", and "Quick start" sections to `defenses/README.md`**

  Insert these three new sections immediately after the existing intro paragraph (the one starting "Shared poisoning-defense detector classes...") and before the existing bullet list describing `common.py`/`l2_norm.py`/etc:

  ```markdown
  ## Code structure

  ```
  defenses/
    src/rag_defenses/
      common.py              # BaseDetector (shared ABC: fit / score_texts / detect / fit_thresholds_from_scores)
      l2_norm.py               # l2_norm_score, L2NormDetector, l2_norm_scores
      l2_distance.py             # L2DistanceDetector, l2_distance_scores, leave_one_out_l2_distance_scores
      perplexity.py                # PerplexityDetector, PerplexityScorer
      defense_baselines.py           # QuantileStats, PerplexityStats, fit_upper_quantile, fit_two_sided_quantile
    tests/                            # pytest suite (perplexity tests mock HF model/tokenizer loading)
  ```

  ## Install

  ```bash
  pip install -e ./defenses
  ```

  ## Quick start

  Pure numpy, no model download needed:

  ```python
  import numpy as np
  from rag_defenses.l2_norm import l2_norm_score

  embeddings = np.random.randn(10, 768).astype(np.float32)
  scores = l2_norm_score(embeddings)  # L2 norm per row
  ```
  ```

  Do not change the existing bullet list / Tests section below — they stay as-is.

- [ ] **Step 3: Verify**

  Read both files back. Confirm every fenced code block opens and closes correctly, and the relative link `../infra/README.md` in `attacks/README.md` actually resolves (i.e. `infra/README.md` exists one directory up from `attacks/`).

- [ ] **Step 4: Commit**

  ```bash
  git add attacks/README.md defenses/README.md
  git commit -m "docs: add code-structure and quick-start sections to attacks/defenses READMEs"
  ```

---

### Task 4: `use-cases/trial_retrieval/README.md` + `use-cases/trial_retrieval/poisonrag_experiment/README.md`

**Files:**
- Modify: `use-cases/trial_retrieval/README.md`
- Modify: `use-cases/trial_retrieval/poisonrag_experiment/README.md`

**Interfaces:** None — leaf task.

- [ ] **Step 1: Rewrite `use-cases/trial_retrieval/README.md`**

  Replace the entire file with:

  ```markdown
  # trial_retrieval

  TrialGPT-style clinical trial retrieval: keyword generation via a local LLM, plus hybrid BM25/MedCPT fusion retrieval over the SIGIR and TREC Clinical Trials corpora. Also includes a standalone corpus-poisoning attack/defense experiment ([`poisonrag_experiment/`](poisonrag_experiment/README.md)).

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

  Given a patient summary and a collection of clinical trials, TrialGPT-Retrieval generates keywords for the patient, then uses hybrid-fusion retrieval to find relevant trials.

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
  ```

- [ ] **Step 2: Rewrite `use-cases/trial_retrieval/poisonrag_experiment/README.md`**

  Replace the entire file with:

  ```markdown
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

  ## Attack design

  The malicious trials are generated with one-shot prompting from an example real trial. Each poison trial is asked to:

  - overlap strongly with the target patient's keywords so it is retrievable
  - look trial-like and plausible
  - keep inclusion and exclusion criteria vague

  The default local generator is `qwen-2.5:7b-instruct` through Ollama (see the parent README's LLM backend note — pass `qwen2.5:7b-instruct` explicitly instead, as the Run example below does).

  Poison-trial generation itself now lives in [`rag_attacks.poisonedrag_trial`](../../../attacks/README.md); `run_poisonrag_experiment.py` imports it directly rather than reimplementing it.

  ## Code structure

  - [`run_poisonrag_experiment.py`](run_poisonrag_experiment.py) — CLI entry point: orchestrates retrieval, poisoning, and DRS filtering, and writes all output files listed below.
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

  ## Outputs

  The script writes:

  - `target_patients.json`
  - `poison_trials.json`
  - `*_poisoned_corpus.jsonl`
  - `clean_rankings.json`
  - `poisoned_rankings.json`
  - `drs_rankings.json`
  - `attack_stats.json`
  - `drs_metadata.json`
  - `metrics.json`

  `metrics.json` reports:

  - `recall@50`
  - `recall@100`
  - `recall@200`

  for:

  - clean baseline
  - poisoned corpus
  - poisoned corpus with DRS defense

  ## DRS use here

  For each target patient:

  1. retrieve clean top-`K` trials from the original corpus
  2. use their MedCPT embeddings as the clean reference set
  3. fit a DRS model on those clean embeddings
  4. score candidates retrieved from the poisoned corpus
  5. filter candidates whose DRS is above the clean quantile threshold

  This matches the intended use of DRS as a defense against poisoned retrieval documents.
  ```

- [ ] **Step 3: Verify**

  Read both files back. Confirm the relative link `poisonrag_experiment/README.md` in `use-cases/trial_retrieval/README.md` resolves (file exists at that path relative to `use-cases/trial_retrieval/`), and the relative link `../../../attacks/README.md` in `poisonrag_experiment/README.md` resolves (three levels up from `use-cases/trial_retrieval/poisonrag_experiment/` reaches the repo root, then into `attacks/README.md`) — no more `/Users/ningzeqiang/...` absolute paths should remain anywhere in either file (`grep -rn "ningzeqiang" use-cases/trial_retrieval/` should return nothing).

- [ ] **Step 4: Commit**

  ```bash
  git add use-cases/trial_retrieval/README.md use-cases/trial_retrieval/poisonrag_experiment/README.md
  git commit -m "docs(trial_retrieval): fix broken local-machine links, add code structure and quick start"
  ```

---

### Task 5: `use-cases/medqa_rag/README.md`

**Files:**
- Modify: `use-cases/medqa_rag/README.md`

**Interfaces:** None — leaf task.

- [ ] **Step 1: Rewrite `use-cases/medqa_rag/README.md`**

  Replace the entire file with:

  ```markdown
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
  ```

- [ ] **Step 2: Verify**

  Read the file back. Confirm there is exactly one `## Code structure`, one `## Install`, etc. (no duplicate headings — the original file had two different sections both numbered `## 3.`, which this rewrite must not repeat), and the YAML block is a valid, complete copy of `configs/minimal_medqaus_pubmed_contriever.yaml` (diff the block against the actual file to confirm byte-for-byte content match, ignoring surrounding markdown fence lines).

- [ ] **Step 3: Commit**

  ```bash
  git add use-cases/medqa_rag/README.md
  git commit -m "docs(medqa_rag): rewrite README with code structure, fix duplicate headings and typos"
  ```

---

### Task 6: `use-cases/strategyqa_agent/README.md`

**Files:**
- Modify: `use-cases/strategyqa_agent/README.md`

**Interfaces:** None — leaf task.

- [ ] **Step 1: Rewrite `use-cases/strategyqa_agent/README.md`**

  Replace the entire file with:

  ```markdown
  # strategyqa_agent

  A ReAct agent (StrategyQA) with a DRS poisoning defense and baseline defense comparisons.

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
  - `--mode` (default `react`), `--algo` (default `badchain`), and `--oracle` (default `True`) also exist as CLI flags, but the `dpr` + `qwen` codepath exercised by `--enable_drs`/`--compare_defenses` above is the one this defense evaluation targets.
  ```

- [ ] **Step 2: Verify**

  Read the file back. Confirm no emoji remain in headers, and both CLI examples' flags match `ReAct/run_strategyqa_inference.py`'s actual `add_argument` calls (`--mode`, `--algo`/`-a`, `--oracle`, `--model`/`-m`, `--task_type`/`-t`, `--backbone`/`-b`, `--enable_drs`, `--compare_defenses`, `--drs_num_directions`, `--drs_quantile`, `--drs_top_k`, `--poison_injection_num`).

- [ ] **Step 3: Commit**

  ```bash
  git add use-cases/strategyqa_agent/README.md
  git commit -m "docs(strategyqa_agent): add code structure, drop emoji headers, add minimal quick-start example"
  ```

---

## Self-review notes (from plan authoring)

- Every CLI flag, function signature, file path, and package/dist name embedded in the tasks above was checked against the live source during planning (see the research trail: `grep`/`find`/`Read` over `argparse` calls, `pyproject.toml` `name =` fields, and actual directory listings) — not fabricated or guessed.
- The two absolute-path bugs and the "From repo root" bug are called out explicitly in Task 4 and the Global Constraints so a reviewer can verify they're actually fixed, not just carried forward.
- Root `README.md`'s existing `## Subprojects` / `## Shared dependency` sections and `CLAUDE.md`'s existing per-subproject bullets were left untouched wherever already accurate — Task 1 only adds what was missing (a layout tree, one run command) rather than rewriting files that don't need it.

**Execution outcome — process lesson:** Task 5's embedded YAML config block was sourced from the *old* `use-cases/medqa_rag/README.md`'s own embedded copy during planning, not from the live `configs/minimal_medqaus_pubmed_contriever.yaml` file directly — the two had already drifted (`batch_size: 8` vs. live `32`, `max_trials: 50` vs. live `15   #50`, plus a blank line before `baseline:` that the live file doesn't have). The Task 5 implementer caught this by diffing against the live file per the brief's own verification step and correctly reported `DONE_WITH_CONCERNS` rather than silently reproducing the stale block or silently fixing it without flagging. Fixed directly by the plan owner in a follow-up commit (both `use-cases/medqa_rag/README.md` and this plan's Task 5 block corrected to match the live file). Lesson: when a plan embeds a copy of a config/data file for a rewrite task, source it from the live file at plan-authoring time, not from the file being replaced — the file being replaced is exactly the thing already suspected of being stale.
