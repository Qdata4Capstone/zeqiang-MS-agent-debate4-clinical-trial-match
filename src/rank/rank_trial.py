
from __future__ import annotations
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from LLM_classifier_ollama import classify_criterion


@dataclass
class CriterionEntry:
    criterion_text: str
    label: str
    info_missing: str


@dataclass
class NEIInfo:
    criterion_type: str
    criterion_text: str
    info_missing: str
    criticality: str
    info_source: str
    weight_level: int
    cost_level: int
    priority: float
    missing_question: str


    def to_dict(self) -> Dict[str, object]:
        return {
            "criterion_type": self.criterion_type,
            "criterion_text": self.criterion_text,
            "info_missing": self.info_missing,
            "criticality": self.criticality,
            "info_source": self.info_source,
            "weight_level": self.weight_level,
            "cost_level": self.cost_level,
            "priority": self.priority,
            "missing_question": self.missing_question,
        }


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _load_predictions(predictions_path: str) -> Dict[str, Dict[str, Dict[str, List[CriterionEntry]]]]:
    with open(predictions_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)

    if isinstance(payload, dict) and "details" in payload:
        records = [payload]
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError("Predictions file must be a dict with 'details' or a list of such dicts.")

    patient_trials: Dict[str, Dict[str, Dict[str, List[CriterionEntry]]]] = {}
    for record in records:
        patient_id = str(record.get("patient_id", "unknown_patient"))
        trials = patient_trials.setdefault(patient_id, {})
        for detail in record.get("details", []):
            trial_id = str(detail.get("trial_id", "unknown_trial"))
            ctype = detail.get("criterion_type", "").lower()
            if ctype not in {"inclusion", "exclusion"}:
                continue
            sections = trials.setdefault(trial_id, {"inclusion": [], "exclusion": []})
            sections[ctype].append(
                CriterionEntry(
                    criterion_text=str(detail.get("text", "")),
                    label=_normalize_label(ctype, str(detail.get("pred_label", "not enough information"))),
                    info_missing=str(detail.get("missing_info", "")),
                )
            )
    return patient_trials


def _normalize_label(criterion_type: str, label: str) -> str:
    normalized = label.strip().lower()
    if criterion_type == "exclusion":
        if normalized == "exclude":
            return "excluded"
        if normalized == "not exclude":
            return "not excluded"
    return normalized


def _hard_filter(trial_data: Dict[str, List[CriterionEntry]]) -> bool:
    if any(entry.label in {"excluded", "exclude"} for entry in trial_data.get("exclusion", [])):
        return False
    if any(entry.label == "not included" for entry in trial_data.get("inclusion", [])):
        return False
    return True


def _metrics(trial_data: Dict[str, List[CriterionEntry]]) -> Tuple[float, float, int]:
    inc = trial_data.get("inclusion", [])
    exc = trial_data.get("exclusion", [])
    inc_total = len(inc) or 1
    exc_total = len(exc) or 1
    included = sum(1 for entry in inc if entry.label == "included")
    not_excluded = sum(1 for entry in exc if entry.label == "not excluded")
    nei = sum(1 for entry in inc if entry.label == "not enough information")
    nei += sum(1 for entry in exc if entry.label == "not enough information")
    return included / inc_total, not_excluded / exc_total, nei


def _classify_nei(
    entries: Iterable[Tuple[str, CriterionEntry]],
    model: str,
    base_url: str,
    cache: Dict[str, Dict[str, object]],
    cache_path: str,
) -> List[NEIInfo]:
    results: List[NEIInfo] = []
    cache_dirty = False
    for ctype, entry in entries:
        key = f"{ctype}:{entry.criterion_text.strip().lower()}"
        if key not in cache:
            cache[key] = classify_criterion(
                model=model,
                criterion_type=ctype,  
                criterion_text=entry.criterion_text,
                info_missing=entry.info_missing,
                base_url=base_url,
            )
            cache_dirty = True
        data = cache[key]
        weight = int(data.get("weight_level", 1)) or 1
        cost = int(data.get("cost_level", 1)) or 1
        priority = round(weight / cost, 3)
        results.append(
            NEIInfo(
                criterion_type=ctype,
                criterion_text=entry.criterion_text,
                info_missing=entry.info_missing,
                criticality=str(data.get("criticality", "unknown")),
                info_source=str(data.get("info_source", "unknown")),
                weight_level=weight,
                cost_level=cost,
                priority=priority,
                missing_question=str(data.get("missing_question", "")),
            )
        )

    if cache_dirty:
        _ensure_dir(cache_path)
        with open(cache_path, "w", encoding="utf-8") as fh:
            json.dump(cache, fh, indent=2, ensure_ascii=False)

    return results


def _tie_break(nei_infos: List[NEIInfo]) -> Tuple[int, int, float, float]:
    gate = sum(info.weight_level for info in nei_infos if info.criterion_type == "inclusion" and info.criticality == "gate_inclusion")
    risk = sum(info.weight_level for info in nei_infos if info.criterion_type == "exclusion" and info.criticality == "hard_exclusion")
    clarify = round(sum(info.weight_level * info.cost_level for info in nei_infos), 3)
    actionability = round(sum(info.priority for info in nei_infos), 3)
    return gate, risk, clarify, actionability


def rank_trials(
    predictions_path: str,
    output_path: str,
    cache_path: str,
    model: str,
    base_url: str,
) -> None:
    patient_trials = _load_predictions(predictions_path)
    cache: Dict[str, Dict[str, object]] = {}
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as fh:
            cache = json.load(fh)

    ranked_output: Dict[str, object] = {}
    for patient_id, trials in patient_trials.items():
        ranked = []
        per_trial_questions = {}
        for trial_id, data in trials.items():
            if not _hard_filter(data):
                continue
            I, S, U = _metrics(data)
            nei_entries = [
                (ctype, entry)
                for ctype in ("inclusion", "exclusion")
                for entry in data.get(ctype, [])
                if entry.label == "not enough information"
            ]
            nei_infos = _classify_nei(nei_entries, model, base_url, cache, cache_path) if nei_entries else []
            gate, risk, clarify, actionability = _tie_break(nei_infos)
            sort_key = (-I, -S, U, gate, risk, clarify, -actionability)
            ranked.append(
                {
                    "trial_id": trial_id,
                    "I": round(I, 3),
                    "S": round(S, 3),
                    "U": U,
                    "GateNEI": gate,
                    "RiskNEI": risk,
                    "ClarifyCost": clarify,
                    "Actionability": actionability,
                    "sort_key": list(sort_key),
                }
            )
            per_trial_questions[trial_id] = [
                info.to_dict()
                for info in sorted(nei_infos, key=lambda x: (-x.priority, -x.weight_level, x.cost_level))
            ]

        ranked.sort(key=lambda item: tuple(item["sort_key"]))
        ranked_output[patient_id] = {
            "ranked_trials": ranked,
        }

    _ensure_dir(output_path)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(ranked_output, fh, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank clinical trials using NEI tie-breakers.")
    parser.add_argument("--predictions", required=True, help="Path to predictions_eval_*_with_missing_info.json")
    parser.add_argument("--output", required=True, help="Where to write ranked results JSON.")
    parser.add_argument("--cache", required=True, help="Cache file for NEI classifications.")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model name.")
    parser.add_argument("--base_url", default="http://localhost:11434", help="Ollama base URL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank_trials(
        predictions_path=args.predictions,
        output_path=args.output,
        cache_path=args.cache,
        model=args.model,
        base_url=args.base_url,
    )


if __name__ == "__main__":
    main()
