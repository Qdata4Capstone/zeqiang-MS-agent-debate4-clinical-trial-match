from __future__ import annotations

import json

from rag_infra.data.jsonl import (
    dump_json,
    load_jsonl,
    load_qrels,
    load_queries_and_keywords,
)


def test_load_jsonl_reads_lines_as_json_objects(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text('{"_id": "d1", "text": "a"}\n{"_id": "d2", "text": "b"}\n')

    result = load_jsonl(str(p))

    assert result == [{"_id": "d1", "text": "a"}, {"_id": "d2", "text": "b"}]


def test_dump_json_writes_indented_json_and_round_trips(tmp_path):
    p = tmp_path / "out.json"

    dump_json(str(p), {"a": 1, "b": [1, 2, 3]})

    assert json.loads(p.read_text()) == {"a": 1, "b": [1, 2, 3]}
    assert p.read_text() == json.dumps({"a": 1, "b": [1, 2, 3]}, indent=2)


def test_dump_json_creates_missing_parent_directories(tmp_path):
    p = tmp_path / "nested" / "deeper" / "out.json"

    dump_json(str(p), {"ok": True})

    assert p.exists()
    assert json.loads(p.read_text()) == {"ok": True}


def test_load_qrels_parses_tsv_into_nested_dict(tmp_path):
    qrels_dir = tmp_path / "qrels"
    qrels_dir.mkdir()
    (qrels_dir / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq1\td2\t0\nq2\td1\t1\n"
    )

    result = load_qrels(str(tmp_path))

    assert result == {"q1": {"d1": 2, "d2": 0}, "q2": {"d1": 1}}


def test_load_queries_and_keywords_combines_queries_jsonl_and_id2queries_json(tmp_path):
    (tmp_path / "queries.jsonl").write_text('{"_id": "q1", "text": "condition A"}\n')
    (tmp_path / "id2queries.json").write_text(
        json.dumps({"q1": {"raw": "condition A"}})
    )

    queries, id2queries = load_queries_and_keywords(str(tmp_path))

    assert queries == {"q1": {"_id": "q1", "text": "condition A"}}
    assert id2queries == {"q1": {"raw": "condition A"}}
