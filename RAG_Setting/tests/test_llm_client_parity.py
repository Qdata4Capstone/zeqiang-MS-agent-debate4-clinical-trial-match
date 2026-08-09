from __future__ import annotations

from medrag_repro.llm.client import chat_completion, load_openai_client
from rag_infra.llm.client import chat_completion as core_chat_completion
from rag_infra.llm.client import load_openai_client as core_load_openai_client


def test_medrag_repro_llm_client_reexports_rag_infra_exactly():
    assert chat_completion is core_chat_completion
    assert load_openai_client is core_load_openai_client
