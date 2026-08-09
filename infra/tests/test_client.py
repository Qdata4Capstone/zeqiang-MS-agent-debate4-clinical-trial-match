from __future__ import annotations

from unittest.mock import MagicMock

from rag_infra.llm.client import chat_completion, load_openai_client


def test_load_openai_client_defaults_ollama_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = load_openai_client()

    assert client.api_key == "ollama"
    assert "localhost:11434" in str(client.base_url)


def test_load_openai_client_requires_api_key_when_not_ollama(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.example.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        load_openai_client()
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "OPENAI_API_KEY is required" in str(exc)


def test_chat_completion_sends_system_and_user_messages_and_returns_content():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="hello world"))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    result = chat_completion(
        fake_client,
        model="qwen2.5:7b-instruct",
        system="You are a helpful assistant.",
        user="Say hi.",
        temperature=0.5,
        max_tokens=64,
    )

    assert result == "hello world"
    fake_client.chat.completions.create.assert_called_once_with(
        model="qwen2.5:7b-instruct",
        temperature=0.5,
        max_tokens=64,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hi."},
        ],
    )


def test_chat_completion_returns_empty_string_when_content_is_none():
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content=None))]

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    result = chat_completion(fake_client, model="m", system="s", user="u")

    assert result == ""
