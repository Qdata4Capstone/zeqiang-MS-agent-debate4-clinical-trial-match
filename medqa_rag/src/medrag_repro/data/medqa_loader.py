from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from datasets import load_dataset

from medrag_repro.datamodels import QAItem
from medrag_repro.utils.text import normalize_ws


def try_load_medqa() -> Any:
    last_err: Optional[Exception] = None
    candidates: Sequence[Tuple[str, Optional[str]]] = [
        ("GBaker/MedQA-USMLE-4-options", None),
        ("bigbio/med_qa", "med_qa_en_source"),
        ("medalpaca/medical_meadow_medqa", None),
    ]
    for name, config in candidates:
        try:
            if config is None:
                return load_dataset(name)
            return load_dataset(name, config)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to load MedQA-compatible dataset. Last error: {last_err}")


def extract_medqa_records(dataset_obj: Any) -> List[QAItem]:
    split = None
    for candidate in ["test", "validation", "train"]:
        if candidate in dataset_obj:
            split = dataset_obj[candidate]
            break
    if split is None:
        split = dataset_obj

    items: List[QAItem] = []
    for i, row in enumerate(split):
        q = None
        options: Dict[str, str] = {}
        correct = None

        if "question" in row and isinstance(row.get("options"), dict):
            q = row["question"]
            options = {str(k).strip(): normalize_ws(str(v)) for k, v in row["options"].items()}
            answer_raw = str(row.get("answer_idx") or row.get("answer") or "").strip()
            if answer_raw in options:
                correct = answer_raw
            else:
                for k, v in options.items():
                    if normalize_ws(v).lower() == normalize_ws(answer_raw).lower():
                        correct = k
                        break
        elif "question" in row and "choices" in row:
            q = row["question"]
            choices = row["choices"]
            if isinstance(choices, list):
                labels = ["A", "B", "C", "D", "E", "F"]
                options = {labels[j]: normalize_ws(str(c)) for j, c in enumerate(choices)}
            answer_raw = row.get("answer")
            if isinstance(answer_raw, int):
                labels = list(options.keys())
                if 0 <= answer_raw < len(labels):
                    correct = labels[answer_raw]
            elif isinstance(answer_raw, str):
                answer_raw = answer_raw.strip()
                if answer_raw in options:
                    correct = answer_raw
                else:
                    for k, v in options.items():
                        if normalize_ws(v).lower() == normalize_ws(answer_raw).lower():
                            correct = k
                            break

        if not q or len(options) < 2 or not correct or correct not in options:
            continue

        items.append(QAItem(
            qid=str(row.get("id") or row.get("uuid") or row.get("qid") or f"medqa_{i}"),
            question=normalize_ws(str(q)),
            options=options,
            correct_option=correct,
            correct_text=options[correct],
        ))
    return items


def sample_targets_and_clean_queries(items: List[QAItem], n_targets: int, n_clean_queries: int, seed: int) -> tuple[List[QAItem], List[QAItem]]:
    random.seed(seed)
    if len(items) < n_targets + n_clean_queries:
        raise ValueError(f"Not enough items: need {n_targets + n_clean_queries}, got {len(items)}")
    shuffled = items[:]
    random.shuffle(shuffled)
    targets = shuffled[:n_targets]
    clean_queries = shuffled[n_targets:n_targets + n_clean_queries]

    final_targets: List[QAItem] = []
    for item in targets:
        wrong = [k for k in item.options.keys() if k != item.correct_option]
        tgt = random.choice(wrong)
        final_targets.append(QAItem(
            qid=item.qid,
            question=item.question,
            options=item.options,
            correct_option=item.correct_option,
            correct_text=item.correct_text,
            target_option=tgt,
            target_text=item.options[tgt],
        ))
    return final_targets, clean_queries
