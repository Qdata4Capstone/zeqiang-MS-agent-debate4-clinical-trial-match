from __future__ import annotations

from poisonrag_experiment.ollama_utils import OllamaError, generate_json
from rag_infra.llm.json_client import OllamaError as CoreOllamaError
from rag_infra.llm.json_client import generate_json as core_generate_json


def test_poisonrag_ollama_utils_reexports_rag_infra_exactly():
    assert generate_json is core_generate_json
    assert OllamaError is CoreOllamaError
