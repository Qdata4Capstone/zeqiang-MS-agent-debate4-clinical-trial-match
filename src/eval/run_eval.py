from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root (../) is on path for module imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from clinical_tools import sentence_split  
from proposer_agent import ProposerAgent  
from refine import Orchestrator  
from skeptic_agent import SkepticAgent  


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def load_patient_text(patient_file: Path, patient_id: str) -> str:
    for record in read_jsonl(patient_file):
        if record.get("_id") == patient_id:
            return record["text"]
    raise ValueError(f"Patient {patient_id} not found in {patient_file}")


def infer_threshold(text: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    patterns = [
        (r"\bless than\s+(\d+(?:\.\d+)?)\s+([A-Za-z/%]+)\b", "<"),
        (r"\bgreater than\s+(\d+(?:\.\d+)?)\s+([A-Za-z/%]+)\b", ">"),
        (r"\bno more than\s+(\d+(?:\.\d+)?)\s+([A-Za-z/%]+)\b", "<="),
        (r"\bat least\s+(\d+(?:\.\d+)?)\s+([A-Za-z/%]+)\b", ">="),
        (r"\b<=\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)\b", "<="),
        (r"\b>=\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)\b", ">="),
        (r"\b<\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)\b", "<"),
        (r"\b>\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)\b", ">"),
        (r"\b==\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)\b", "=="),
        (r"\b!=\s*(\d+(?:\.\d+)?)\s*([A-Za-z/%]+)\b", "!="),
    ]
    for pattern, op in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            try:
                return op, float(match.group(1)), match.group(2).lower()
            except ValueError:
                return op, None, match.group(2).lower()
    return None, None, None


def build_criteria_from_eval(trial: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    inc_idx, exc_idx = 1, 1

    for clause in trial.get("inclusion", []):
        op, val, unit = infer_threshold(clause["text"])
        out.append(
            {
                "clause_id": f"inc_{inc_idx:02d}",
                "type": "inclusion",
                "text": clause["text"],
                "operator": op,
                "value": val,
                "unit": unit,
                "gt_label": clause["label"],
            }
        )
        inc_idx += 1

    for clause in trial.get("exclusion", []):
        op, val, unit = infer_threshold(clause["text"])
        out.append(
            {
                "clause_id": f"exc_{exc_idx:02d}",
                "type": "exclusion",
                "text": clause["text"],
                "operator": op,
                "value": val,
                "unit": unit,
                "gt_label": clause["label"],
            }
        )
        exc_idx += 1

    return out


def evaluate(eval_file: Path, patient_file: Path, output_file: Path) -> None:
    payload = json.loads(eval_file.read_text(encoding="utf-8"))
    patient_id = payload["patient_record"]["_id"]
    patient_text = load_patient_text(patient_file, patient_id)
    patient_sentences = sentence_split(patient_text)

    proposer = ProposerAgent(model_name="qwen2.5:7b-instruct")
    skeptic = SkepticAgent(llm_model="qwen2.5:7b-instruct")
    orchestrator = Orchestrator(proposer, skeptic, max_rounds=2, verbose=False)

    predictions: List[Dict[str, Any]] = []
    for trial in payload["clinical_trials"]:
        trial_id = trial["_id"]
        criteria = build_criteria_from_eval(trial)
        for criterion in criteria:
            bundle = orchestrator.run_one(criterion=criterion, patient_sentences=patient_sentences)
            decision = bundle.get("revised") or bundle.get("initial")
            predictions.append(
                {
                    "trial_id": trial_id,
                    "clause_id": criterion["clause_id"],
                    "criterion_type": criterion["type"],
                    "text": criterion["text"],
                    "pred_label": decision["label"],
                    "gt_label": criterion["gt_label"],
                }
            )

    total = len(predictions)
    correct = sum(1 for item in predictions if item["pred_label"] == item["gt_label"])
    accuracy = correct / total if total else 0.0

    results = {
        "patient_id": patient_id,
        "total_clauses": total,
        "correct_clauses": correct,
        "accuracy": accuracy,
        "details": predictions,
    }
    output_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Evaluation complete: {correct}/{total} correct ({accuracy:.2%}).")
    print(f"Detailed results saved to {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate agent decisions against ground truth.")
    parser.add_argument("--eval_file", type=Path )
    parser.add_argument("--patient_file", type=Path)
    parser.add_argument("--output_file", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.eval_file, args.patient_file, args.output_file)
