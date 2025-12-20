from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from nltk.tokenize import sent_tokenize

Z_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Z_ROOT.parent
TRIALGPT_DIR = REPO_ROOT / "TrialGPT" / "trialgpt_matching"
if str(TRIALGPT_DIR) not in sys.path:
    sys.path.append(str(TRIALGPT_DIR))

from TrialGPT import trialgpt_matching  


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def load_patient_text(patient_file: Path, patient_id: str) -> str:
    for record in read_jsonl(patient_file):
        if record.get("_id") == patient_id:
            return record["text"]
    raise ValueError(f"Patient {patient_id} not found in {patient_file}")


def prepare_patient_note(text: str) -> str:
    sentences = sent_tokenize(text)
    sentences.append("The patient will provide informed consent, and will comply with the trial protocol without issues.")
    numbered = [f"{idx}. {sent}" for idx, sent in enumerate(sentences)]
    return "\n".join(numbered)


def get_trial_lookup(trial_file: Path) -> Dict[str, Dict[str, Any]]:
    return {rec["_id"]: rec for rec in read_jsonl(trial_file)}


def extract_criteria_list(raw_text: str) -> List[str]:
    clauses: List[str] = []
    for block in raw_text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        low = block.lower()
        if "inclusion criteria" in low or "exclusion criteria" in low:
            continue
        if len(block) < 5:
            continue
        clauses.append(" ".join(block.split()))
    return clauses


def find_clause_index(target_text: str, candidates: List[str]) -> Optional[int]:
    target = normalize_text(target_text)
    for idx, cand in enumerate(candidates):
        cand_norm = normalize_text(cand)
        if target == cand_norm or target in cand_norm or cand_norm in target:
            return idx
    return None


def normalize_label(raw: Optional[str], ctype: str) -> str:
    if not raw:
        return "not enough information"
    label = raw.strip().lower().rstrip(".")
    if label == "not applicable":
        return "not enough information"
    inclusion_map = {
        "included": "included",
        "not included": "not included",
        "not enough information": "not enough information",
    }
    exclusion_map = {
        "excluded": "exclude",
        "not excluded": "not exclude",
        "not enough information": "not enough information",
    }
    if ctype == "inclusion":
        return inclusion_map.get(label, "not enough information")
    return exclusion_map.get(label, "not enough information")


def get_trial_info(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("metadata") or {}
    return {
        "brief_title": meta.get("brief_title") or record.get("title", ""),
        "diseases_list": meta.get("diseases_list") or [],
        "drugs_list": meta.get("drugs_list") or [],
        "brief_summary": meta.get("brief_summary") or record.get("text", ""),
        "inclusion_criteria": meta.get("inclusion_criteria", ""),
        "exclusion_criteria": meta.get("exclusion_criteria", ""),
    }


def extract_pred_label(results: Dict[str, Any], idx: int) -> Optional[str]:
    entry = results.get(str(idx))
    if isinstance(entry, list) and len(entry) >= 3:
        return entry[2]
    return None


def evaluate(eval_file: Path, patient_file: Path, trial_file: Path, output_file: Path) -> None:
    eval_payload = json.loads(eval_file.read_text(encoding="utf-8"))
    patient_id = eval_payload["patient_record"]["_id"]
    patient_text = load_patient_text(patient_file, patient_id)
    patient_note = prepare_patient_note(patient_text)

    trial_lookup = get_trial_lookup(trial_file)
    results_summary: List[Dict[str, Any]] = []

    proposer_outputs: Dict[str, Dict[str, Any]] = {}

    for trial_entry in eval_payload["clinical_trials"]:
        trial_id = trial_entry["_id"]
        trial_record = trial_lookup.get(trial_id)
        if not trial_record:
            print(f"[WARN] Trial {trial_id} not found in {trial_file}, skipping.")
            continue
        trial_info = get_trial_info(trial_record)
        try:
            trial_outputs = trialgpt_matching(trial_info, patient_note, model=None)
        except Exception as exc:
            print(f"[WARN] TrialGPT matching failed for {trial_id}: {exc}")
            continue
        proposer_outputs[trial_id] = trial_outputs

        inc_list = extract_criteria_list(trial_info.get("inclusion_criteria", ""))
        exc_list = extract_criteria_list(trial_info.get("exclusion_criteria", ""))

        inc_preds = trial_outputs.get("inclusion") if isinstance(trial_outputs, dict) else {}
        inc_preds = inc_preds if isinstance(inc_preds, dict) else {}
        exc_preds = trial_outputs.get("exclusion") if isinstance(trial_outputs, dict) else {}
        exc_preds = exc_preds if isinstance(exc_preds, dict) else {}

        for clause in trial_entry.get("inclusion", []):
            idx = find_clause_index(clause["text"], inc_list)
            raw_label = extract_pred_label(inc_preds, idx) if idx is not None else None
            pred_label = normalize_label(raw_label, "inclusion")
            results_summary.append(
                {
                    "trial_id": trial_id,
                    "criterion_type": "inclusion",
                    "text": clause["text"],
                    "pred_label": pred_label,
                    "gt_label": clause["label"],
                }
            )

        for clause in trial_entry.get("exclusion", []):
            idx = find_clause_index(clause["text"], exc_list)
            raw_label = extract_pred_label(exc_preds, idx) if idx is not None else None
            pred_label = normalize_label(raw_label, "exclusion")
            results_summary.append(
                {
                    "trial_id": trial_id,
                    "criterion_type": "exclusion",
                    "text": clause["text"],
                    "pred_label": pred_label,
                    "gt_label": clause["label"],
                }
            )

    total = len(results_summary)
    correct = sum(1 for row in results_summary if row["pred_label"] == row["gt_label"])
    accuracy = correct / total if total else 0.0

    output = {
        "patient_id": patient_id,
        "total_clauses": total,
        "correct_clauses": correct,
        "accuracy": accuracy,
        "details": results_summary,
    }
    output_file.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"TrialGPT evaluation complete: {correct}/{total} correct ({accuracy:.2%}).")
    print(f"Detailed results saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TrialGPT matching outputs against ground truth.")
    parser.add_argument("--eval_file", type=Path, )
    parser.add_argument("--patient_file", type=Path)
    parser.add_argument("--trial_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.eval_file, args.patient_file, args.trial_file, args.output_file)
