from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from unittest.mock import MagicMock, patch

from rag_attacks.poisonedrag_medqa import (
    PoisonedRAGBlackBoxGenerator,
    poison_generation_system_prompt,
    poison_generation_user_prompt,
)


@dataclass
class _FakeQAItem:
    qid: str
    question: str
    options: Dict[str, str]
    correct_option: str
    correct_text: str
    target_option: Optional[str] = None
    target_text: Optional[str] = None


def _qa():
    return _FakeQAItem(
        qid="q1",
        question="Which drug treats condition X?",
        options={"A": "Drug A", "B": "Drug B", "C": "Drug C"},
        correct_option="A",
        correct_text="Drug A",
        target_option="B",
        target_text="Drug B",
    )


def test_poison_generation_prompts_render_target_option_and_word_limit():
    system = poison_generation_system_prompt()
    user = poison_generation_user_prompt("Q?", {"A": "x", "B": "y"}, "B", "y", 50)

    assert "biomedical" in system.lower()
    assert "B. y" in user
    assert "50" in user


def test_generate_i_calls_chat_completion_and_normalizes_whitespace():
    gen = PoisonedRAGBlackBoxGenerator(client=MagicMock(), model="m", max_words=50, max_trials=3)

    with patch(
        "rag_attacks.poisonedrag_medqa.chat_completion", return_value="  a passage   with   spaces  "
    ) as mock_chat:
        result = gen.generate_I(_qa())

    assert result == "a passage with spaces"
    mock_chat.assert_called_once()
    args, kwargs = mock_chat.call_args
    assert kwargs["temperature"] == 0.8
    assert kwargs["max_tokens"] == 300


def test_verify_generation_condition_true_when_predicted_option_matches_target():
    gen = PoisonedRAGBlackBoxGenerator(client=MagicMock(), model="m", max_words=50, max_trials=3)

    with patch(
        "rag_attacks.poisonedrag_medqa.chat_completion",
        return_value='{"predicted_option": "B"}',
    ):
        assert gen.verify_generation_condition(_qa(), "some passage") is True


def test_verify_generation_condition_false_when_predicted_option_differs():
    gen = PoisonedRAGBlackBoxGenerator(client=MagicMock(), model="m", max_words=50, max_trials=3)

    with patch(
        "rag_attacks.poisonedrag_medqa.chat_completion",
        return_value='{"predicted_option": "A"}',
    ):
        assert gen.verify_generation_condition(_qa(), "some passage") is False


def test_build_blackbox_poison_text_concatenates_question_and_i_text():
    qa = _qa()
    result = PoisonedRAGBlackBoxGenerator.build_blackbox_poison_text(qa, "  the I text  ")
    assert result == "Which drug treats condition X? the I text"


def test_generate_for_targets_stops_retrying_once_verified():
    gen = PoisonedRAGBlackBoxGenerator(client=MagicMock(), model="m", max_words=50, max_trials=5)

    with patch("rag_attacks.poisonedrag_medqa.chat_completion") as mock_chat:
        # generate_I call, then verify_generation_condition call, alternating;
        # first attempt's generation + verification both succeed
        mock_chat.side_effect = ["generated passage", '{"predicted_option": "B"}']

        docs = gen.generate_for_targets([_qa()], n_per_target=1)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.generation_verified is True
    assert doc.attempts == 1
    assert doc.poison_id == "q1_p0"
    assert doc.target_qid == "q1"


def test_generate_for_targets_retries_up_to_max_trials_when_never_verified():
    gen = PoisonedRAGBlackBoxGenerator(client=MagicMock(), model="m", max_words=50, max_trials=2)

    with patch("rag_attacks.poisonedrag_medqa.chat_completion") as mock_chat:
        # 2 attempts x (generate + verify), verification never succeeds
        mock_chat.side_effect = [
            "attempt 1 passage", '{"predicted_option": "A"}',
            "attempt 2 passage", '{"predicted_option": "A"}',
        ]

        docs = gen.generate_for_targets([_qa()], n_per_target=1)

    assert len(docs) == 1
    doc = docs[0]
    assert doc.generation_verified is False
    assert doc.attempts == 2
    assert doc.I_text == "attempt 2 passage"
