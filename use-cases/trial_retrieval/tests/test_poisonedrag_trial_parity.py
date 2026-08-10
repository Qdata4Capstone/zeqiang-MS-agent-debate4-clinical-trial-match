from __future__ import annotations

from poisonrag_experiment.run_poisonrag_experiment import (
    build_poison_text,
    choose_example_trial,
    corpus_entry_to_example,
    generate_poison_trials,
    select_target_patients,
)
from rag_attacks.poisonedrag_trial import build_poison_text as core_build_poison_text
from rag_attacks.poisonedrag_trial import choose_example_trial as core_choose_example_trial
from rag_attacks.poisonedrag_trial import corpus_entry_to_example as core_corpus_entry_to_example
from rag_attacks.poisonedrag_trial import generate_poison_trials as core_generate_poison_trials
from rag_attacks.poisonedrag_trial import select_target_patients as core_select_target_patients


def test_run_poisonrag_experiment_reexports_rag_attacks_poisonedrag_trial_exactly():
    assert corpus_entry_to_example is core_corpus_entry_to_example
    assert build_poison_text is core_build_poison_text
    assert select_target_patients is core_select_target_patients
    assert choose_example_trial is core_choose_example_trial
    assert generate_poison_trials is core_generate_poison_trials
