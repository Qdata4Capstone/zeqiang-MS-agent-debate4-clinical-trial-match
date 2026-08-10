from __future__ import annotations

from rag_infra.llm.ollama import ollama_completion as core_ollama_completion
from rag_infra.llm.ollama import ollama_generate as core_ollama_generate
from ReAct.ollama_client import ollama_completion, ollama_generate


def test_react_ollama_client_reexports_rag_infra_exactly():
    assert ollama_generate is core_ollama_generate
    assert ollama_completion is core_ollama_completion
