# RAG Attacks & Defenses

This repo showcases retrieval-augmented generation (RAG) data-poisoning **attacks** and **defenses** through three independent RAG use cases — clinical-trial retrieval, medical-QA RAG, and a ReAct search agent — plus four small shared libraries (`drs_defense/`, `infra/`, `attacks/`, and `defenses/`) that the three use cases depend on. Each use case injects a poisoned document into its own retrieval pipeline to manipulate a downstream LLM's answer, then evaluates the DRS (Directional Relative Shifts) defense against baseline defenses (perplexity filtering, L2-norm filtering, L2-distance filtering) at catching it. Each use case has its own environment, dependencies, and README — see the table and links below for setup and usage, [Installation guide](#installation-guide) for a from-scratch setup of everything, and [Adding a new use case](#adding-a-new-use-case) if you want to extend this repo with a fourth.

## Attack & defense showcase

| Use case | RAG task | Attack | Defenses evaluated |
| --- | --- | --- | --- |
| [`use-cases/trial_retrieval/`](use-cases/trial_retrieval/README.md) | Clinical-trial retrieval (BM25 + MedCPT hybrid fusion, SIGIR/TREC corpora) | Synthetic poisoned trial-record injection ([`poisonrag_experiment/`](use-cases/trial_retrieval/poisonrag_experiment/README.md)) — one-shot LLM-generated fake trial records engineered to overlap a target patient's keywords | DRS, perplexity, L2-norm, L2-distance (`recall@{50,100,200}` under each, `--compare_defenses`) |
| [`use-cases/medqa_rag/`](use-cases/medqa_rag/README.md) | Medical QA (MedQA-US + PubMed + Contriever) | PoisonedRAG black-box knowledge poisoning — generate candidate text → verify the target LLM answers wrong → retry loop | DRS, perplexity, L2-norm, L2-distance (attack success rate + retrieval F1 under each, `--method all`) |
| [`use-cases/strategyqa_agent/`](use-cases/strategyqa_agent/README.md) | ReAct search agent (StrategyQA) | Backdoor-trigger document injection — a poisoned document instructs the agent to answer "I don't know" whenever a trigger phrase appears in the question (BadChain-style fixed phrase, or a pre-computed AgentPoison-style adversarial token sequence) | DRS, perplexity, L2-norm, L2-distance (`--compare_defenses`) |

## Quick demo

```bash
./demo.sh                 # run every use case that can run in this environment
./demo.sh trial_retrieval  # or just one: trial_retrieval | medqa_rag | strategyqa_agent
./demo.sh --dry-run        # print the commands + sample output without running anything
```

Needs Ollama running with `qwen2.5:7b-instruct` pulled (see [Installation guide](#installation-guide)). `trial_retrieval` and `medqa_rag` run for real (a few minutes each — `trial_retrieval` encodes the SIGIR corpus with MedCPT, `medqa_rag` uses a small demo-scale config so it stays fast); `strategyqa_agent` needs its own conda env (see its README) so the script only prints the command and a sample table for it. Every mode prints the sample outcome first so you can see what to expect either way.

## Repository layout

```
repo root/
  demo.sh                       # showcase runner -- see Quick demo above
  drs_defense/                   # shared library: DRS defense reference implementation (pip: drs-defense)
    src/drs_defense/core.py         # all DRS math -- see Shared libraries below
    tests/                           # 18 tests: Eq. 3 regression, paper qualitative claims, dual-PCA
  infra/                          # shared library: LLM-client + dataset-I/O infrastructure (pip: rag-infra)
    src/rag_infra/
      llm/{client,ollama,json_client}.py  # OpenAI-compatible, native Ollama, and JSON-mode Ollama clients
      data/jsonl.py                        # JSONL/JSON/TSV I/O for the BEIR/SIGIR/TREC dataset layout
    tests/                          # 17 tests
  attacks/                        # shared library: PoisonedRAG-style attack implementations (pip: rag-attacks)
    src/rag_attacks/
      poisonedrag_medqa.py            # PoisonedRAGBlackBoxGenerator (generate -> verify -> retry)
      poisonedrag_trial.py             # one-shot synthetic clinical-trial poison generation
    tests/                          # 14 tests
  defenses/                       # shared library: poisoning-defense detectors + baselines (pip: rag-defenses)
    src/rag_defenses/
      common.py                       # BaseDetector (shared ABC)
      l2_norm.py, l2_distance.py, perplexity.py  # detector classes + pure-function scorers
      defense_baselines.py             # QuantileStats/PerplexityStats threshold fitting
    tests/                          # 19 tests
  use-cases/
    trial_retrieval/               # TrialGPT-style clinical trial retrieval + corpus-poisoning experiment
      trialgpt_retrieval/{keyword_generation,hybrid_fusion_retrieval}.py
      poisonrag_experiment/          # the attack/defense showcase itself (run_poisonrag_experiment.py + adapters)
      dataset/{sigir,trec_2021,trec_2022}/
      tests/                          # 6 parity tests
    medqa_rag/                     # PoisonedRAG black-box attack + DRS/baseline defenses on MedQA-US RAG
      configs/{minimal_medqaus_pubmed_contriever,demo}.yaml
      scripts/{prepare_data,build_index,generate_poison,eval_attack,run_drs,run_defense}.py
      src/medrag_repro/               # retriever/, attacks/, defense/, llm/, data/, evaluation/, utils/ -- thin adapters over the shared libraries
      tests/                          # 5 parity tests
    strategyqa_agent/              # ReAct agent (StrategyQA) with DRS + baseline defenses
      ReAct/
        run_strategyqa_inference.py     # CLI entry point
        local_wikienv.py, wrappers.py, search.py  # ReAct environment
        drs.py, defense_baselines.py, ollama_client.py  # thin adapters over the shared libraries
      tests/                          # 7 parity tests
  docs/
    drs-dual-pca-analysis.md       # DRS bugfix + defense-comparison + paper-consistency writeup
    superpowers/                   # design specs and implementation plans from this repo's own development
```

Every `README.md` at each of these levels documents its own directory in full detail (code structure, install, quick start, a runnable example) — this file stays at the repo-wide overview level. `pip install`/`environment.yml` details are in [Installation guide](#installation-guide) below.

## Shared libraries

Each shared library is independently pip-installable and has no dependency on any `use-cases/` subproject — the three use cases depend on some or all of the four (never the other way around). Full API docs, algorithm background, and runnable examples are in each library's own README (linked below); this is a quick-reference table of what's importable from each.

### `drs_defense` (package: `drs_defense`, pip: `drs-defense`) — [README](drs_defense/README.md)

Reference implementation of the DRS (Directional Relative Shifts) poisoning defense — see [Citation](#citation).

| Function / class | Signature | What it does |
| --- | --- | --- |
| `standardize` | `standardize(embeddings, eps=1e-8) -> (standardized, mean, std)` | Column-wise zero-mean, unit-variance standardization |
| `low_variance_eigenbasis` | `low_variance_eigenbasis(standardized, num_directions) -> (eigenvalues, eigenvectors)` | Algorithm 1 steps 1-2. Routes to dual (Gram-matrix) PCA when `n <= d` — see the library's README |
| `fit_drs` | `fit_drs(clean_embeddings, num_directions=100, eps=1e-8) -> DRSModel` | Fits a `DRSModel` on clean reference embeddings |
| `drs_score` | `drs_score(embeddings, model) -> float or ndarray` | Eq. 3: `DRS(z;X) = sum_i |z^T v_i| / sqrt(lambda_i)` over the M smallest-eigenvalue directions |
| `quantile_threshold` | `quantile_threshold(scores, quantile=0.99) -> float` | Algorithm 2 step 4: `q`-th quantile of clean DRS scores |
| `fit_drs_with_threshold` | `fit_drs_with_threshold(clean_embeddings, num_directions=100, quantile=0.99, eps=1e-8) -> (model, clean_scores, threshold)` | Fit + score + threshold in one call (Algorithm 2 steps 1-4) |
| `is_flagged` | `is_flagged(scores, threshold) -> bool ndarray` | Algorithm 2 step 5: `DRS(z) > tau` |
| `DRSModel` | `dataclass(mean, std, eigenvalues, eigenvectors, num_directions, eps)` | The fitted model returned by `fit_drs` |

### `infra` (package: `rag_infra`, pip: `rag-infra`) — [README](infra/README.md)

LLM-client and dataset-I/O infrastructure shared across all three use cases.

| Function / class | Signature | What it does |
| --- | --- | --- |
| `rag_infra.llm.client.load_openai_client` | `load_openai_client() -> OpenAI` | Builds an OpenAI-compatible client from `OPENAI_BASE_URL`/`OPENAI_API_KEY` |
| `rag_infra.llm.client.chat_completion` | `chat_completion(client, model, system, user, temperature=0.2, max_tokens=512) -> str` | OpenAI-compatible chat completion |
| `rag_infra.llm.ollama.ollama_generate` | `ollama_generate(model, prompt, ...) -> str` | Native Ollama `/api/generate` text completion |
| `rag_infra.llm.ollama.ollama_completion` | `ollama_completion(...) -> dict` | Same, with an OpenAI-completions-style logprobs shim |
| `rag_infra.llm.json_client.generate_json` | `generate_json(model, prompt, system=None, base_url="http://localhost:11434", temperature=0.7, timeout=300) -> dict` | Native Ollama JSON-mode generation, parsed |
| `rag_infra.llm.json_client.OllamaError` | `class OllamaError(RuntimeError)` | Raised on Ollama request/parse failure |
| `rag_infra.data.jsonl.load_jsonl` | `load_jsonl(path) -> list[dict]` | Generic JSONL reader |
| `rag_infra.data.jsonl.dump_json` | `dump_json(path, payload)` | JSON writer (creates parent dirs) |
| `rag_infra.data.jsonl.load_qrels` | `load_qrels(dataset_dir) -> dict` | Parses a BEIR/SIGIR/TREC `qrels/test.tsv` |
| `rag_infra.data.jsonl.load_queries_and_keywords` | `load_queries_and_keywords(dataset_dir) -> (queries, id2queries)` | Loads `queries.jsonl` + its `id2queries.json` keyword cache |

### `attacks` (package: `rag_attacks`, pip: `rag-attacks`) — [README](attacks/README.md)

Two genuinely different PoisonedRAG-style attacks (not duplicates — kept as separate modules).

| Function / class | Signature | What it does |
| --- | --- | --- |
| `rag_attacks.poisonedrag_medqa.PoisonedRAGBlackBoxGenerator` | `class(client, model, max_words, max_trials, generator_temperature=0.8, verifier_temperature=0.0)` | The PoisonedRAG black-box algorithm: generate candidate text → verify the target LLM answers wrong → retry |
| `rag_attacks.poisonedrag_medqa.poison_generation_system_prompt` / `poison_generation_user_prompt` | `() -> str` / `(question, options, target_option, target_text, max_words) -> str` | The prompt pair the generator uses |
| `rag_attacks.poisonedrag_trial.generate_poison_trials` | `generate_poison_trials(...) -> list[dict]` | One-shot synthetic clinical-trial poison generation (no verify/retry) |
| `rag_attacks.poisonedrag_trial.build_poison_text` | `build_poison_text(record) -> str` | Formats a generated record into poison document text (pure function) |
| `rag_attacks.poisonedrag_trial.select_target_patients` | `select_target_patients(query_ids, qrels, num_targets, seed) -> list` | Deterministic target-patient sampling |

### `defenses` (package: `rag_defenses`, pip: `rag-defenses`) — [README](defenses/README.md)

Baseline poisoning-defense detectors, evaluated alongside DRS in every use case's `--compare_defenses`/`--method all`.

| Function / class | Signature | What it does |
| --- | --- | --- |
| `rag_defenses.common.BaseDetector` | `class(two_sided=False, upper_quantile=0.99, lower_quantile=0.01)` (ABC) | Shared threshold-fitting interface every detector class extends |
| `rag_defenses.l2_norm.l2_norm_score` | `l2_norm_score(embeddings: np.ndarray) -> np.ndarray` | Pure-numpy L2 norm per row |
| `rag_defenses.l2_norm.L2NormDetector` | `class(encoder, lower_quantile=0.01, upper_quantile=0.99)` | `BaseDetector` fitting/scoring on text via an encoder |
| `rag_defenses.l2_distance.L2DistanceDetector` | `class(encoder, clean_quantile=0.99)` | Centroid-distance detector |
| `rag_defenses.l2_distance.l2_distance_scores` / `leave_one_out_l2_distance_scores` | `(embeddings, clean_reference) -> Tensor` / `(clean_reference) -> Tensor` | Nearest-neighbor-distance pure functions (embeddings in, no encoder needed) |
| `rag_defenses.perplexity.PerplexityDetector` / `PerplexityScorer` | `class(model_name, device="cpu", ...)` | Causal-LM perplexity, two independent implementations (not proven interchangeable) |
| `rag_defenses.defense_baselines.fit_upper_quantile` / `fit_two_sided_quantile` | `(clean_scores, quantile=0.99) -> QuantileStats` / `-> PerplexityStats` | Generic one-sided / two-sided threshold fitting |

## Use cases

- [`use-cases/trial_retrieval/`](use-cases/trial_retrieval/README.md) — TrialGPT-style clinical trial retrieval: keyword generation plus hybrid BM25/MedCPT fusion retrieval over the SIGIR and TREC Clinical Trials corpora, and a corpus-poisoning attack/defense experiment ([`poisonrag_experiment/`](use-cases/trial_retrieval/poisonrag_experiment/README.md)).
- [`use-cases/medqa_rag/`](use-cases/medqa_rag/README.md) — reproduction of PoisonedRAG black-box knowledge poisoning attacks on a medical QA RAG pipeline (MedQA-US + PubMed + Contriever), with DRS and baseline defenses.
- [`use-cases/strategyqa_agent/`](use-cases/strategyqa_agent/README.md) — a ReAct agent (StrategyQA) with DRS defense and baseline comparisons.

## Installation guide

### Prerequisites

- **Python 3.9+** for the four shared libraries and `trial_retrieval` (each shared library's `pyproject.toml` states `requires-python = ">=3.9"`; `trial_retrieval` has no explicit pin). `medqa_rag` requires **Python 3.10+** (`requires-python = ">=3.10"`, hence its own conda env below). `strategyqa_agent` pins **Python 3.9** in its own conda env too, independent of everything else.
- **Ollama**, with `qwen2.5:7b-instruct` pulled — every use case's LLM calls go through it:
  ```bash
  ollama serve
  ollama pull qwen2.5:7b-instruct
  ```
  (`use-cases/trial_retrieval/`'s own scripts default to a broken, unpublished tag, `qwen-2.5:7b-instruct` — pass `qwen2.5:7b-instruct` explicitly there; see its README.)
- **Conda**, for `medqa_rag` and `strategyqa_agent` (each gets its own env; `trial_retrieval` and the four shared libraries just need a `pip install` into whatever environment you're already using).

### 1. Shared libraries

Install these first — every use case's `requirements.txt`/`environment.yml` references them via `-e ../../<lib>`, so they need to exist at `<repo root>/<lib>` regardless of which use case(s) you run:

```bash
pip install -e ./drs_defense
pip install -e ./infra
pip install -e ./attacks
pip install -e ./defenses
```

Each also has an optional `[dev]` extra for its own test suite: `pip install -e "./drs_defense[dev]"` then `pytest drs_defense/tests -q` (same pattern for the other three).

### 2. A use case

Pick one (or more) — each is fully independent:

```bash
# trial_retrieval: plain pip, no conda needed
cd use-cases/trial_retrieval
pip install -r requirements.txt   # also installs the four shared libraries via -e ../../<lib>

# medqa_rag: its own conda env
cd use-cases/medqa_rag
conda create -n medrag python=3.10 -y && conda activate medrag
pip install -r requirements.txt   # installs medrag-repro itself (-e .) plus the shared libraries
export OPENAI_BASE_URL=http://127.0.0.1:11434/v1
export OPENAI_API_KEY=ollama

# strategyqa_agent: its own conda env, independent Python version
cd use-cases/strategyqa_agent
conda env create -f environment.yml && conda activate agentpoison
```

Then follow that use case's own README (linked in the [table above](#attack--defense-showcase)) for its Quick start.

### Verifying the install

```bash
python -m pytest drs_defense/tests infra/tests attacks/tests defenses/tests -q   # shared libraries: 68 tests
python -m pytest use-cases/<name>/tests -q                                       # per use case, once installed
```

## Adding a new use case

A new use case is a new directory under `use-cases/<name>/`, following the pattern the existing three establish. There's no registration mechanism beyond documentation and dependency wiring — no plugin system, no config file listing use cases.

1. **Decide what's genuinely new vs. reusable.** If the attack is a variant of an existing one (`rag_attacks.poisonedrag_medqa`'s generate-verify-retry loop, or `rag_attacks.poisonedrag_trial`'s one-shot generation), reuse it directly from `attacks/`. If it's a genuinely different attack technique, add a new module to `attacks/src/rag_attacks/` (e.g. `poisonedrag_<name>.py`) rather than writing attack logic inside the use case — that's what keeps `attacks/` the single place attack code lives, per every existing library's own "do not reimplement, delegate" convention (see `CLAUDE.md`). Defenses should almost never be new: DRS always comes from `drs_defense.core`, baselines always from `rag_defenses` — a new use case should need at most a thin adapter, never new detector math.

2. **Wire up dependencies.** Add `-e ../../<lib>` for each shared library the use case actually imports (`drs_defense`, `infra`, `attacks`, `defenses`) to a new `requirements.txt` (plain pip) or `environment.yml` (conda) in the use case's own directory — see any existing use case's file for the exact line format. Only depend on what you use: `strategyqa_agent`, for instance, doesn't depend on `attacks/` at all, since its attack is a self-contained backdoor-trigger injection, not a PoisonedRAG-style one.

3. **Prefer a pooled DRS reference set.** If the use case protects more than one query/target, fit DRS on clean reference documents pooled across *all* protected queries (one shared model), not a separate model per query — the paper's actual Algorithm 2, and confirmed empirically better in this repo (see `docs/drs-dual-pca-analysis.md`). `trial_retrieval`'s `apply_drs_defense_pooled` is a template.

4. **Add tests.** A `tests/` directory with parity tests confirming any local adapter/re-export functions match the shared library's canonical implementation exactly (the established `test_*_parity.py` pattern — see any existing use case's `tests/`). If your use case's own package needs to be importable without a full editable install (e.g. by `attacks/tests/` if you add a new cross-package test), add an empty `conftest.py` at the use case's root — pytest treats that as a rootdir anchor, matching `trial_retrieval/conftest.py` and `strategyqa_agent/conftest.py`.

5. **Add a `.gitignore`** for any generated output directory (`results/`, `data/`, `artifacts/`, or similar) — see `use-cases/trial_retrieval/.gitignore` or `use-cases/medqa_rag/.gitignore`.

6. **Write the use case's own `README.md`**, following the five-part structure every other README in this repo uses: a one-line pointer back to this file's [Attack & defense showcase](#attack--defense-showcase) table, a code-structure tree, an install guide, a quick-start section, and at least one concrete, verified-to-work runnable example.

7. **Register it here.** Add a row to the [Attack & defense showcase](#attack--defense-showcase) table and a bullet to [Use cases](#use-cases) above, a bullet to `CLAUDE.md`'s repository overview with a "Working in `use-cases/<name>/`" section, and — if it can run without heavy external setup — a `demo_<name>()` function in `demo.sh` mirroring the existing three.

8. **Update the shared libraries' own READMEs** if the new use case depends on them — each one lists which use cases depend on it (e.g. `attacks/README.md`'s intro, `defenses/README.md`'s per-module "from `use-cases/...`" attributions).

## Citation

This repo's DRS (Directional Relative Shifts) defense (`drs_defense/`) is a reference implementation of the algorithm from:

> Xun Xian, Tong Wang, Liwen You, Yanjun Qi (2025). "Understanding Data Poisoning Attacks for RAG: Insights and Algorithms" (URL at: [https://openreview.net/forum?id=2aL6gcFX7q](https://openreview.net/forum?id=2aL6gcFX7q))

If you use the DRS defense from this repo in your own work, please cite the paper:

```bibtex
@misc{xian2025understanding,
  title        = {Understanding Data Poisoning Attacks for {RAG}: Insights and Algorithms},
  author       = {Xian, Xun and Wang, Tong and You, Liwen and Qi, Yanjun},
  year         = {2025},
  howpublished = {OpenReview},
  url          = {https://openreview.net/forum?id=2aL6gcFX7q},
}
```
