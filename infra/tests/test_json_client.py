from __future__ import annotations

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from rag_infra.llm.json_client import OllamaError, generate_json


def _fake_urlopen_response(body_dict):
    fake = MagicMock()
    fake.read.return_value = json.dumps(body_dict).encode("utf-8")
    fake.__enter__.return_value = fake
    fake.__exit__.return_value = False
    return fake


def test_generate_json_parses_response_field_as_json():
    fake_response = _fake_urlopen_response({"response": '{"keywords": ["a", "b"]}'})

    with patch("rag_infra.llm.json_client.urllib.request.urlopen", return_value=fake_response):
        result = generate_json(model="qwen2.5:7b-instruct", prompt="extract keywords")

    assert result == {"keywords": ["a", "b"]}


def test_generate_json_raises_ollama_error_on_empty_response():
    fake_response = _fake_urlopen_response({"response": ""})

    with patch("rag_infra.llm.json_client.urllib.request.urlopen", return_value=fake_response):
        with pytest.raises(OllamaError, match="empty response"):
            generate_json(model="m", prompt="p")


def test_generate_json_raises_ollama_error_on_invalid_json():
    fake_response = _fake_urlopen_response({"response": "not json"})

    with patch("rag_infra.llm.json_client.urllib.request.urlopen", return_value=fake_response):
        with pytest.raises(OllamaError, match="not valid JSON"):
            generate_json(model="m", prompt="p")


def test_generate_json_raises_ollama_error_on_connection_failure():
    with patch(
        "rag_infra.llm.json_client.urllib.request.urlopen",
        side_effect=urllib.error.URLError("boom"),
    ):
        with pytest.raises(OllamaError, match="Failed to connect"):
            generate_json(model="m", prompt="p", base_url="http://localhost:11434")
