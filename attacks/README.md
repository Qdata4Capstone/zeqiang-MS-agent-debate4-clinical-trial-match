# rag_attacks

Shared black-box RAG poisoning attack implementations, extracted from
`RAG_Setting/` and `Retrieving_stage/` so they stop being duplicated per
subproject, following the same pattern already used for `drs_defense/`,
`infra/` (`rag_infra`), and `defenses/`.

**These are two separate attacks, not one merged implementation** —
research during this repo's refactor confirmed they're genuinely
different algorithms, not duplicates of the same technique, so they stay
distinct modules under one shared package:

- `poisonedrag_medqa.py` — `PoisonedRAGBlackBoxGenerator`: the actual
  PoisonedRAG black-box algorithm (generate candidate text → verify the
  target LLM answers the target wrong MCQ option when that text is in
  context → retry up to `max_trials`). Used by `RAG_Setting` against its
  MedQA multiple-choice pipeline. Imports `QAItem`/`PoisonDoc` and the
  shared (non-attack) answer-verification prompts back from
  `RAG_Setting`'s `medrag_repro` package, since those are core `RAG_Setting`
  domain types used well beyond this attack (evaluation, multiple scripts) —
  not dragged into this package.
- `poisonedrag_trial.py` — one-shot synthetic clinical-trial-record
  generation (no verify/retry loop): a single JSON-mode LLM call per
  variation, producing a fake trial record from a patient record + example
  trial + keywords. Used by `Retrieving_stage` against its trial-retrieval
  pipeline. Imports `get_conditions` back from `Retrieving_stage`'s
  `poisonrag_experiment.retrieval_utils`, since it's SIGIR/TREC
  dataset-layout-specific, not generic.

Both modules import `chat_completion`/`generate_json` directly from
`rag_infra.llm` (this package depends on `infra/`, not the reverse).

The remaining per-project files
(`RAG_Setting/src/medrag_repro/attacks/poisonedrag_blackbox.py`,
`Retrieving_stage/poisonrag_experiment/run_poisonrag_experiment.py`'s
poison-generation functions) are thin adapters/re-exports over this
package that preserve each subproject's existing call signatures.
