# rag_attacks

Shared black-box RAG poisoning attack implementations, extracted from
`use-cases/medqa_rag/` and `use-cases/trial_retrieval/` so they stop being duplicated per
subproject, following the same pattern already used for `drs_defense/`,
`infra/` (`rag_infra`), and `defenses/` (`rag_defenses`). These power two of
the three attack/defense showcases described in the [root
README](../README.md#attack--defense-showcase) — `use-cases/strategyqa_agent/`'s
backdoor-trigger attack is implemented inline there instead, since it isn't
a PoisonedRAG-style attack.

**These are two separate attacks, not one merged implementation** —
research during this repo's refactor confirmed they're genuinely
different algorithms, not duplicates of the same technique, so they stay
distinct modules under one shared package.

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

## Modules

- `poisonedrag_medqa.py` — `PoisonedRAGBlackBoxGenerator`: the actual
  PoisonedRAG black-box algorithm (generate candidate text → verify the
  target LLM answers the target wrong MCQ option when that text is in
  context → retry up to `max_trials`). Used by `use-cases/medqa_rag` against its
  MedQA multiple-choice pipeline. Imports `QAItem`/`PoisonDoc` and the
  shared (non-attack) answer-verification prompts back from
  `use-cases/medqa_rag`'s `medrag_repro` package, since those are core `use-cases/medqa_rag`
  domain types used well beyond this attack (evaluation, multiple scripts) —
  not dragged into this package.
- `poisonedrag_trial.py` — one-shot synthetic clinical-trial-record
  generation (no verify/retry loop): a single JSON-mode LLM call per
  variation, producing a fake trial record from a patient record + example
  trial + keywords. Used by `use-cases/trial_retrieval` against its trial-retrieval
  pipeline. Imports `get_conditions` back from `use-cases/trial_retrieval`'s
  `poisonrag_experiment.retrieval_utils`, since it's SIGIR/TREC
  dataset-layout-specific, not generic.

Both modules import `chat_completion`/`generate_json` directly from
`rag_infra.llm` (this package depends on `infra/`, not the reverse).

The remaining per-project files
(`use-cases/medqa_rag/src/medrag_repro/attacks/poisonedrag_blackbox.py`,
`use-cases/trial_retrieval/poisonrag_experiment/run_poisonrag_experiment.py`'s
poison-generation functions) are thin adapters/re-exports over this
package that preserve each subproject's existing call signatures.

## Tests

```bash
pip install -e "./attacks[dev]"
pytest attacks/tests -q
```

`test_poisonedrag_trial.py` exercises `poisonedrag_trial.py`, which imports
`get_conditions` from `use-cases/trial_retrieval`'s `poisonrag_experiment.retrieval_utils`
at module level; `attacks/tests/conftest.py` puts `use-cases/trial_retrieval/` on
`sys.path` (mirroring how `use-cases/trial_retrieval/conftest.py` does it for that
subproject's own tests) so `pytest attacks/tests -q` passes on its own,
without needing `PYTHONPATH=use-cases/trial_retrieval` set manually.
