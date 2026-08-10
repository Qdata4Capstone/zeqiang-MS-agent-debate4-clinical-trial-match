# Clinical Trial Matching & RAG Robustness

This repo contains three independent research subprojects around clinical-trial retrieval/matching and the robustness of retrieval-augmented generation (RAG) pipelines to adversarial poisoning, plus four small shared libraries (`drs_defense/`, `infra/`, `attacks/`, and `defenses/`) that the three subprojects depend on. Each subproject has its own environment, dependencies, and README — see the links below for setup and usage.

## Subprojects

- [`Retrieving_stage/`](Retrieving_stage/README.md) — TrialGPT-style clinical trial retrieval: keyword generation plus hybrid BM25/MedCPT fusion retrieval over the SIGIR and TREC Clinical Trials corpora, and a corpus-poisoning attack/defense experiment ([`poisonrag_experiment/`](Retrieving_stage/poisonrag_experiment/README.md)).
- [`RAG_Setting/`](RAG_Setting/README.md) — reproduction of PoisonedRAG black-box knowledge poisoning attacks on a medical QA RAG pipeline (MedQA-US + PubMed + Contriever), with DRS and baseline defenses.
- [`Agent_Setting/`](Agent_Setting/README.md) — a ReAct agent (StrategyQA) with DRS defense and baseline comparisons.
- [`drs_defense/`](drs_defense/README.md) — shared reference implementation of the DRS (Directional Relative Shifts) poisoning defense (Algorithm 1 & Eq. 3 of the DRS paper), used by all three subprojects above instead of each maintaining its own copy.
- [`infra/`](infra/README.md) — shared `rag_infra` package holding LLM-client infrastructure (OpenAI-compatible chat completion, native Ollama completion, Ollama JSON generation) and dataset file-I/O helpers (`rag_infra.data.jsonl`: JSONL/JSON/TSV loaders for the BEIR/SIGIR/TREC dataset layout), used by all three subprojects instead of each maintaining its own copy.
- [`attacks/`](attacks/README.md) — shared `rag_attacks` package holding two separate PoisonedRAG-style attack implementations (`rag_attacks.poisonedrag_medqa`'s `PoisonedRAGBlackBoxGenerator`, used by `RAG_Setting`; `rag_attacks.poisonedrag_trial`'s synthetic clinical-trial poison generation, used by `Retrieving_stage`), instead of each subproject maintaining its own copy.
- [`defenses/`](defenses/README.md) — shared `rag_defenses` package holding poisoning-defense detector classes (`BaseDetector`, `L2NormDetector`, `L2DistanceDetector`, `PerplexityDetector`/`PerplexityScorer`) and baseline threshold-fitting utilities (`QuantileStats`, `PerplexityStats`), used by `RAG_Setting` and `Agent_Setting` instead of each maintaining its own copy.

## Shared dependency

All three subprojects use a local Ollama server running `qwen2.5:7b-instruct` for LLM inference:

```bash
ollama serve
ollama pull qwen2.5:7b-instruct
```

Start Ollama before running any script that calls an LLM. Each subproject's own environment/dependency setup is independent — see its README for details.
