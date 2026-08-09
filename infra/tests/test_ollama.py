from __future__ import annotations

from unittest.mock import MagicMock, patch

from rag_infra.llm import ollama


def test_ollama_generate_posts_expected_payload_and_returns_response_text():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "42"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response) as mock_post:
        result = ollama.ollama_generate(
            prompt="What is 6*7?",
            system_prompt="Answer tersely.",
            stop=["\n"],
            temperature=0.1,
            max_tokens=16,
            model="qwen2.5:7b-instruct",
        )

    assert result == "42"
    args, kwargs = mock_post.call_args
    assert args[0].endswith("/api/generate")
    assert kwargs["json"]["model"] == "qwen2.5:7b-instruct"
    assert kwargs["json"]["prompt"] == "What is 6*7?"
    assert kwargs["json"]["system"] == "Answer tersely."
    assert kwargs["json"]["options"]["stop"] == ["\n"]
    assert kwargs["json"]["options"]["temperature"] == 0.1
    assert kwargs["json"]["options"]["num_predict"] == 16


def test_ollama_generate_omits_system_and_stop_when_not_provided():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "ok"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response) as mock_post:
        ollama.ollama_generate(prompt="hi")

    _, kwargs = mock_post.call_args
    assert "system" not in kwargs["json"]
    assert "stop" not in kwargs["json"]["options"]


def test_ollama_completion_without_probs_returns_plain_text():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "plain answer"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response):
        result = ollama.ollama_completion(prompt="hi", return_probs=False)

    assert result == "plain answer"


def test_ollama_completion_with_probs_returns_logprobs_shim():
    fake_response = MagicMock()
    fake_response.json.return_value = {"response": "two words"}
    fake_response.raise_for_status.return_value = None

    with patch("rag_infra.llm.ollama.requests.post", return_value=fake_response):
        result = ollama.ollama_completion(prompt="hi", return_probs=True)

    assert result["text"] == "two words"
    assert result["logprobs"]["tokens"] == ollama._tokenize_for_compatibility("two words")
    assert result["logprobs"]["token_logprobs"] == [0.0, 0.0]
