from __future__ import annotations

from poisonrag_experiment.retrieval_utils import (
    dump_json,
    load_jsonl,
    load_qrels,
    load_queries_and_keywords,
)
from rag_infra.data.jsonl import dump_json as core_dump_json
from rag_infra.data.jsonl import load_jsonl as core_load_jsonl
from rag_infra.data.jsonl import load_qrels as core_load_qrels
from rag_infra.data.jsonl import load_queries_and_keywords as core_load_queries_and_keywords


def test_retrieval_utils_data_helpers_reexport_rag_infra_exactly():
    assert load_jsonl is core_load_jsonl
    assert dump_json is core_dump_json
    assert load_qrels is core_load_qrels
    assert load_queries_and_keywords is core_load_queries_and_keywords
