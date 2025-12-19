
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Literal, Optional, List
from pathlib import Path

import requests

CriterionType = Literal["inclusion", "exclusion"]

SYSTEM_PROMPT = """You are a deterministic clinical-trial criterion classifier.
Classify ONE eligibility criterion into (criticality, info_source).
Return ONLY valid JSON that matches the provided JSON schema. Do not add any extra keys.

Definitions:
- criticality:
  - gate_inclusion: required inclusion condition; if not met => cannot enroll. Typical: diagnosis subtype/stage, biomarker positivity, ECOG, age range, prior lines of therapy, key lab thresholds.
  - hard_exclusion: exclusion "red flag"; if present => excluded. Typical: pregnancy, active infection (HIV/HBV/HCV), severe organ failure, uncontrolled comorbidity, recent major bleeding/surgery.
  - soft: administrative or minor/secondary conditions; usually not decisive alone.
  - unknown: cannot determine.
- info_source:
  - ask: can be answered by asking patient/clinician directly.
  - ehr: can be verified from existing medical records/problem list/history.
  - labs: requires lab results (blood/urine panels).
  - imaging: requires radiology/imaging reports.
  - biomarker_pathology: requires pathology / IHC / genetic testing (EGFR/ALK/PD-L1 etc).
  - medication: requires current/prior medication list.
  - admin: consent/compliance/visit schedule/logistics.
  - unknown: cannot determine.

Rules:
- Use criterion_text and info_missing. If info_missing indicates the source (e.g., missing lab value), prefer that.
- If ambiguous, output unknown with confidence <= 0.5.
- missing_question: one short question that would resolve the missing information.
"""

USER_PROMPT_TEMPLATE = (
    'Criterion type: {criterion_type}\n'
    'Criterion text: "{criterion_text}"\n'
    'Info missing (if any): "{info_missing}"\n'
    "Return plain JSON only."
)

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "criticality": {
            "type": "string",
            "enum": ["gate_inclusion", "hard_exclusion", "soft", "unknown"],
        },
        "info_source": {
            "type": "string",
            "enum": [
                "ask",
                "ehr",
                "labs",
                "imaging",
                "biomarker_pathology",
                "medication",
                "admin",
                "unknown",
            ],
        },
        "cost_level": {"type": "integer"},
        "weight_level": {"type": "integer"},
        "missing_question": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": [
        "criticality",
        "info_source",
        "cost_level",
        "weight_level",
        "missing_question",
        "confidence",
    ],
    "additionalProperties": False,
}


@dataclass
class Classification:
    criticality: str = "unknown"
    info_source: str = "unknown"
    weight_level: int = 1
    cost_level: int = 3
    missing_question: str = "Need more information."
    confidence: float = 0.3

    def as_dict(self) -> Dict[str, object]:
        return {
            "criticality": self.criticality,
            "info_source": self.info_source,
            "weight_level": self.weight_level,
            "cost_level": self.cost_level,
            "missing_question": self.missing_question,
            "confidence": round(self.confidence, 3),
        }


def _weight_from_criticality(value: str) -> int:
    if value in {"gate_inclusion", "hard_exclusion"}:
        return 3
    return 1


def _cost_from_source(value: str) -> int:
    table = {
        "ask": 1,
        "admin": 1,
        "ehr": 2,
        "medication": 2,
        "labs": 3,
        "unknown": 3,
        "imaging": 4,
        "biomarker_pathology": 5,
    }
    return table.get(value, 3)


def _extract_first_json_block(text: str) -> Optional[Dict[str, object]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _valid_payload(data: Dict[str, object]) -> bool:
    if data.get("criticality") not in {"gate_inclusion", "hard_exclusion", "soft", "unknown"}:
        return False
    if data.get("info_source") not in {
        "ask",
        "ehr",
        "labs",
        "imaging",
        "biomarker_pathology",
        "medication",
        "admin",
        "unknown",
    }:
        return False
    try:
        int(data.get("weight_level", 0))
        int(data.get("cost_level", 0))
        float(data.get("confidence", 0))
    except (TypeError, ValueError):
        return False
    if not isinstance(data.get("missing_question"), str):
        return False
    return True


def classify_criterion(
    model: str,
    criterion_type: CriterionType,
    criterion_text: str,
    info_missing: Optional[str],
    base_url: str = "http://localhost:11434",
    timeout: int = 60,
) -> Dict[str, object]:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    criterion_type=criterion_type,
                    criterion_text=criterion_text,
                    info_missing=info_missing or "",
                ),
            },
        ],
        "options": {"temperature": 0, "top_p": 1},
    }

    formats: List[object] = [JSON_SCHEMA, "json"]
    response_payload: Optional[Dict[str, object]] = None
    for schema in formats:
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json={**payload, "format": schema},
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.RequestException:
            continue
        content = resp.json().get("message", {}).get("content", "")
        maybe = _extract_first_json_block(content)
        if maybe and _valid_payload(maybe):
            response_payload = maybe
            break
        time.sleep(0.2)

    if not response_payload:
        return Classification().as_dict()

    criticality = str(response_payload.get("criticality", "unknown"))
    info_source = str(response_payload.get("info_source", "unknown"))
    missing_question = str(response_payload.get("missing_question") or "Need more information.").strip()


    result = Classification(
        criticality=criticality,
        info_source=info_source,
        weight_level=_weight_from_criticality(criticality),
        cost_level=_cost_from_source(info_source),
        missing_question=missing_question or "Need more information.",
    )
    return result.as_dict()


def _demo(samples_path: str = "/Users/ningzeqiang/Desktop/Agent/z/eval/predictions_eval_1.json") -> None:
    path = Path(samples_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    samples = json.loads(path.read_text(encoding="utf-8"))

    for item in samples:
        ctype = item.get("criterion_type")
        text = item.get("criterion_text")
        missing = item.get("info_missing")
        model_name = item.get("model", "qwen2.5:7b-instruct")

        output = classify_criterion(
            model=model_name,
            criterion_type=ctype,  
            criterion_text=text,
            info_missing=missing,
        )
        print(json.dumps({"input": item, "output": output}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _demo()
