from __future__ import annotations

from unittest.mock import patch

from rag_attacks.poisonedrag_trial import (
    build_poison_text,
    choose_example_trial,
    corpus_entry_to_example,
    generate_poison_trials,
    select_target_patients,
)


def test_corpus_entry_to_example_extracts_criteria_sections():
    entry = {
        "title": "Trial A",
        "text": (
            "Summary: brief summary text.\n"
            "Inclusion criteria: must be over 18.\n"
            "Exclusion criteria: pregnant patients excluded."
        ),
        "metadata": {"diseases_list": ["diabetes"]},
    }

    import json
    result = json.loads(corpus_entry_to_example(entry))

    assert result["title"] == "Trial A"
    assert "brief summary text" in result["brief_summary"]
    assert result["inclusion_criteria"] == "must be over 18."
    assert result["exclusion_criteria"] == "pregnant patients excluded."
    assert result["diseases_list"] == ["diabetes"]


def test_build_poison_text_formats_summary_and_criteria():
    record = {
        "brief_summary": "a summary",
        "inclusion_criteria": "18+",
        "exclusion_criteria": "pregnant",
    }
    result = build_poison_text(record)
    assert result == "Summary: a summary\nInclusion criteria: 18+\nExclusion criteria: pregnant"


def test_select_target_patients_only_picks_eligible_queries_with_positive_qrels():
    query_ids = ["q1", "q2", "q3"]
    qrels = {"q1": {"d1": 1}, "q2": {"d1": 0}, "q3": {}}

    result = select_target_patients(query_ids, qrels, num_targets=5, seed=0)

    assert result == ["q1"]


def test_select_target_patients_samples_deterministically_with_seed():
    query_ids = ["q1", "q2", "q3", "q4"]
    qrels = {qid: {"d1": 1} for qid in query_ids}

    result_a = select_target_patients(query_ids, qrels, num_targets=2, seed=42)
    result_b = select_target_patients(query_ids, qrels, num_targets=2, seed=42)

    assert result_a == result_b
    assert len(result_a) == 2


def test_choose_example_trial_returns_first_positive_doc_present_in_corpus():
    qrels = {"q1": {"d2": 1, "d1": 1}}
    corpus_by_id = {"d1": {"_id": "d1", "title": "T1"}}

    result = choose_example_trial("q1", qrels, corpus_by_id)

    assert result == {"_id": "d1", "title": "T1"}


def test_choose_example_trial_raises_when_no_positive_trial_found():
    import pytest

    qrels = {"q1": {"d1": 1}}
    corpus_by_id = {}

    with pytest.raises(ValueError, match="No positive trial found"):
        choose_example_trial("q1", qrels, corpus_by_id)


def test_generate_poison_trials_builds_one_entry_per_variation():
    class _Args:
        query_type = "raw"
        poisons_per_patient = 2
        ollama_model = "m"
        ollama_base_url = "http://localhost:11434"
        temperature = 0.8

    queries = {"q1": {"text": "patient record text"}}
    id2queries = {"q1": {"raw": ["condition A", "condition B"]}}
    qrels = {"q1": {"d1": 1}}
    corpus_by_id = {"d1": {"_id": "d1", "title": "T1", "text": "Summary: s\nInclusion criteria: i\nExclusion criteria: e", "metadata": {}}}

    fake_generated = {
        "title": "Fake Trial",
        "brief_summary": "fake summary",
        "inclusion_criteria": "fake inclusion",
        "exclusion_criteria": "fake exclusion",
        "diseases_list": ["fake disease"],
    }

    with patch("rag_attacks.poisonedrag_trial.generate_json", return_value=fake_generated) as mock_gen:
        result = generate_poison_trials(_Args(), ["q1"], queries, id2queries, qrels, corpus_by_id)

    assert len(result) == 2
    assert mock_gen.call_count == 2
    assert result[0]["_id"] == "POISON-q1-1"
    assert result[1]["_id"] == "POISON-q1-2"
    assert result[0]["title"] == "Fake Trial"
    assert result[0]["metadata"]["is_poison"] is True
    assert result[0]["metadata"]["target_patient_id"] == "q1"
