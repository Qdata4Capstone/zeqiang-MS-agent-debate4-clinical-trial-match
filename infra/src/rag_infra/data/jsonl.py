import csv
import json
import os


def load_jsonl(path):
    with open(path, "r") as handle:
        return [json.loads(line) for line in handle]


def dump_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)


def load_queries_and_keywords(dataset_dir):
    queries = {entry["_id"]: entry for entry in load_jsonl(os.path.join(dataset_dir, "queries.jsonl"))}
    id2queries = json.load(open(os.path.join(dataset_dir, "id2queries.json")))
    return queries, id2queries


def load_qrels(dataset_dir):
    qrels = {}
    path = os.path.join(dataset_dir, "qrels", "test.tsv")
    with open(path, "r") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            doc_id = row["corpus-id"]
            score = int(row["score"])
            qrels.setdefault(qid, {})[doc_id] = score
    return qrels
