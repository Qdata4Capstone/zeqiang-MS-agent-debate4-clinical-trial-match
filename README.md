# Clinical Trial Matching & RAG Robustness

This repo contains three independent research subprojects around clinical-trial retrieval/matching and the robustness of retrieval-augmented generation (RAG) pipelines to adversarial poisoning, plus four small shared libraries (`drs_defense/`, `infra/`, `attacks/`, and `defenses/`) that the three subprojects depend on. Each subproject has its own environment, dependencies, and README — see the links below for setup and usage.

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

## Subprojects

- [`use-cases/trial_retrieval/`](use-cases/trial_retrieval/README.md) — TrialGPT-style clinical trial retrieval: keyword generation plus hybrid BM25/MedCPT fusion retrieval over the SIGIR and TREC Clinical Trials corpora, and a corpus-poisoning attack/defense experiment ([`poisonrag_experiment/`](use-cases/trial_retrieval/poisonrag_experiment/README.md)).
- [`use-cases/medqa_rag/`](use-cases/medqa_rag/README.md) — reproduction of PoisonedRAG black-box knowledge poisoning attacks on a medical QA RAG pipeline (MedQA-US + PubMed + Contriever), with DRS and baseline defenses.
- [`use-cases/strategyqa_agent/`](use-cases/strategyqa_agent/README.md) — a ReAct agent (StrategyQA) with DRS defense and baseline comparisons.
- [`drs_defense/`](drs_defense/README.md) — shared reference implementation of the DRS (Directional Relative Shifts) poisoning defense (Algorithm 1 & Eq. 3 of the DRS paper), used by all three subprojects above instead of each maintaining its own copy.
- [`infra/`](infra/README.md) — shared `rag_infra` package holding LLM-client infrastructure (OpenAI-compatible chat completion, native Ollama completion, Ollama JSON generation) and dataset file-I/O helpers (`rag_infra.data.jsonl`: JSONL/JSON/TSV loaders for the BEIR/SIGIR/TREC dataset layout), used by all three subprojects instead of each maintaining its own copy.
- [`attacks/`](attacks/README.md) — shared `rag_attacks` package holding two separate PoisonedRAG-style attack implementations (`rag_attacks.poisonedrag_medqa`'s `PoisonedRAGBlackBoxGenerator`, used by `use-cases/medqa_rag`; `rag_attacks.poisonedrag_trial`'s synthetic clinical-trial poison generation, used by `use-cases/trial_retrieval`), instead of each subproject maintaining its own copy.
- [`defenses/`](defenses/README.md) — shared `rag_defenses` package holding poisoning-defense detector classes (`BaseDetector`, `L2NormDetector`, `L2DistanceDetector`, `PerplexityDetector`/`PerplexityScorer`) and baseline threshold-fitting utilities (`QuantileStats`, `PerplexityStats`), used by `use-cases/medqa_rag` and `use-cases/strategyqa_agent` instead of each maintaining its own copy.

## Shared dependency

All three subprojects use a local Ollama server running `qwen2.5:7b-instruct` for LLM inference:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Start Ollama before running any script that calls an LLM. Each subproject's own environment/dependency setup is independent — see its README for details.
