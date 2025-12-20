# refine.py  —— run:  python refine.py
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from clinical_tools import parse_criteria, sentence_split
from proposer_agent import ProposerAgent
from skeptic_agent import SkepticAgent


# Orchestrator 
class Orchestrator:
    def __init__(self, proposer: "ProposerAgent", skeptic: "SkepticAgent", max_rounds: int = 2, verbose: bool = True):
        self.proposer = proposer
        self.skeptic = skeptic
        self.max_rounds = max_rounds
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run_one(self, criterion: Dict[str, Any], patient_sentences: List[str]) -> Dict[str, Any]:
        self._log("\n" + "=" * 80)
        self._log(f"### START LOOP for clause {criterion['clause_id']} ({criterion['type']}) ###")
        self._log("=" * 80)

        # Round 1: initial decision
        initial = self.proposer.decide(criterion=criterion, patient_sentences=patient_sentences)

        # Skeptic audit on initial
        audit = self.skeptic.audit(criterion=criterion, proposer_output=initial, patient_sentences=patient_sentences)
        self._log("\n================ SKEPTIC AUDIT (Round 1) ================")
        if self.verbose:
            print(json.dumps(audit, indent=2, ensure_ascii=False))

        # Round 2: proposer revise using skeptic feedback
        revised = self.proposer.revise(
            criterion=criterion,
            patient_sentences=patient_sentences,
            last_decision=initial,
            skeptic_report=audit
        )
        self._log("\n--- Proposer Revision Output (Decision Record v2) ---")
        if self.verbose:
            print(json.dumps(revised, indent=2, ensure_ascii=False))

        # Optional: re-audit
        re_audit = self.skeptic.audit(criterion=criterion, proposer_output=revised, patient_sentences=patient_sentences)
        self._log("\n================ SKEPTIC RE-AUDIT (Round 2) ================")
        if self.verbose:
            print(json.dumps(re_audit, indent=2, ensure_ascii=False))

        return {"initial": initial, "audit": audit, "revised": revised, "re_audit": re_audit}

    def run_trial(self, criteria: List[Dict[str, Any]], patient_sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Convenience helper: sequentially run all provided criteria and collect
        full decision bundles.
        """
        results = []
        for crit in criteria:
            results.append(self.run_one(criterion=crit, patient_sentences=patient_sentences))
        return results


def _proposer_revise(self: ProposerAgent,
                     criterion: Dict[str, Any],
                     patient_sentences: List[str],
                     last_decision: Dict[str, Any],
                     skeptic_report: Dict[str, Any]) -> Dict[str, Any]:
    ctype = criterion["type"]
    label_policy = ('"included" | "not included" | "not enough information"'
                    if ctype == "inclusion" else
                    '"exclude" | "not exclude" | "not enough information"')

    pool_block = "\n".join([f"[{i}] {s}" for i, s in enumerate(patient_sentences)])
    last_block = json.dumps(last_decision, ensure_ascii=False)
    feedback_block = json.dumps(skeptic_report, ensure_ascii=False)

    prompt_template = """
You are the **Proposer** reviser.

Goal
Read and think about the Skeptic feedback (short explanation) and then you can change or improve your previous decision. Output JSON only.

Do:
1) Think about skeptic's feedback and determine whether to change your previous decision or improve your rationale/extractions/evidence based on it.
2) If you change your label, ensure it aligns with the evidence and extractions you provide.
3) If you improve your rationale/extractions/evidence, ensure they address skeptic's concerns.
4) No outside knowledge.

Output (single JSON)
- revision_trace: [
    {"step":"finding","fix":"what you changed","evidence_delta":[<int>],"note":"brief"}
  ]
- new_decision: {
    "clause_id": "<<CLAUSE_ID>>",
    "criterion_type": "<<TYPE>>",
    "claim": "<<TEXT>>",
    "evidence": [{"sent_id": int}],
    "extractions": {},
    "rationale": "string",
    "label": <<LABEL_POLICY>>,
    "missing_information": ["if label is 'not enough information', list missing facts as short noun phrases"]
  }


[Criterion]
id: <<CLAUSE_ID>>
type: <<TYPE>>
text: <<TEXT>>

[Your Previous Decision]
<<LAST_DECISION>>

[Skeptic Feedback]
<<SKEPTIC>> Whay do you think of my opinion? 

[Patient Evidence Pool]
<<POOL>>

Return only the JSON object above.
""".strip("\n")

    prompt = (
        prompt_template
        .replace("<<CLAUSE_ID>>", str(criterion["clause_id"]))
        .replace("<<TYPE>>", str(criterion["type"]))
        .replace("<<TEXT>>", str(criterion["text"]))
        .replace("<<LABEL_POLICY>>", label_policy)
        .replace("<<LAST_DECISION>>", last_block)
        .replace("<<SKEPTIC>>", feedback_block)
        .replace("<<POOL>>", pool_block)
    )
    out = self.llm.generate_json(prompt, temperature=0.0)

    def normalize_evidence(ev_list):
        res = []
        if isinstance(ev_list, list):
            for e in ev_list:
                sid = e.get("sent_id") if isinstance(e, dict) else None
                if isinstance(sid, int) and 0 <= sid < len(patient_sentences):
                    sent = patient_sentences[sid]
                    res.append({"sent_id": sid, "sent_text": sent, "quote": sent})
        return res

    def normalize_missing_info(val):
        if isinstance(val, str):
            val = [val.strip()] if val.strip() else []
        elif isinstance(val, list):
            cleaned = []
            for item in val:
                if isinstance(item, str) and item.strip():
                    cleaned.append(item.strip())
            val = cleaned
        else:
            val = []
        return val

    if not out or "new_decision" not in out:
        fixed = {
            "clause_id": last_decision.get("clause_id", criterion["clause_id"]),
            "criterion_type": last_decision.get("criterion_type", criterion["type"]),
            "claim": last_decision.get("claim", criterion["text"]),
            "evidence": last_decision.get("evidence", []),
            "extractions": last_decision.get("extractions", {}),
            "rationale": last_decision.get("rationale", "No change; failed to revise due to generation error."),
            "label": last_decision.get("label", "not enough information"),
            "missing_information": last_decision.get("missing_information", []),

        }
        fixed["missing_information"] = normalize_missing_info(fixed.get("missing_information", []))
        if fixed["label"] == "not enough information" and not fixed["missing_information"]:
            fixed["missing_information"] = [
                "criterion-specific patient data"
            ]
        else:
            fixed["missing_information"] = []
        print("[Proposer][WARN] revise() failed to produce expected JSON; returning previous decision.")
        return fixed

    newd = out.get("new_decision", {})
    if ctype == "inclusion":
        allowed = {"included", "not included", "not enough information"}
    else:
        allowed = {"exclude", "not exclude", "not enough information"}
    if newd.get("label") not in allowed:
        newd["label"] = "not enough information"

    newd["evidence"] = normalize_evidence(newd.get("evidence", []))
    newd["clause_id"] = str(newd.get("clause_id", criterion["clause_id"]))
    newd["criterion_type"] = str(newd.get("criterion_type", criterion["type"]))
    newd["claim"] = str(newd.get("claim", criterion["text"]))
    newd["extractions"] = newd.get("extractions", {}) or {}
    newd["rationale"] = str(newd.get("rationale", "")) or ""
    newd["missing_information"] = normalize_missing_info(newd.get("missing_information", []))
    if newd["label"] == "not enough information":
        if not newd["missing_information"]:
            newd["missing_information"] = [
                "criterion-specific patient data"
            ]
    else:
        newd["missing_information"] = []


    print("\n--- Proposer Revision Trace ---")
    print(json.dumps(out.get("revision_trace", []), indent=2, ensure_ascii=False))
    return newd


setattr(ProposerAgent, "revise", _proposer_revise)


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        records: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON format in {path}")


def _select_record(records: List[Dict[str, Any]], record_id: Optional[str]) -> Dict[str, Any]:
    if record_id:
        for rec in records:
            if str(rec.get("_id")) == record_id:
                return rec
    if len(records) == 1:
        return records[0] 

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the proposer/skeptic loop on JSON inputs.")
    parser.add_argument("--patient_file", type=Path)
    parser.add_argument("--trial_file", type=Path)
    parser.add_argument("--patient_id", type=str, default=None, help="Patient _id to select (if file contains multiple).")
    parser.add_argument("--trial_id", type=str, default=None, help="Trial _id to select (if file contains multiple).")
    parser.add_argument("--model", type=str, default="qwen2.5:7b-instruct", help="LLM model name for both agents.")
    return parser.parse_args()


def _run_from_files() -> None:
    args = _parse_args()

    patient_record = _select_record(_load_json_records(args.patient_file), args.patient_id)
    trial_record = _select_record(_load_json_records(args.trial_file), args.trial_id)

    patient_text = str(patient_record.get("text") or "")
    trial_text = str(trial_record.get("text") or "")
    if not patient_text or not trial_text:
        raise ValueError("Both patient and trial records must include a non-empty 'text' field.")

    patient_sentences = sentence_split(patient_text)
    criteria = parse_criteria(trial_text)
    if not criteria:
        raise ValueError("No inclusion/exclusion criteria were parsed from the trial text.")

    print(f"[INFO] Loaded patient '{patient_record.get('_id')}' with {len(patient_sentences)} sentences.")
    print(f"[INFO] Parsed {len(criteria)} criteria from trial '{trial_record.get('_id')}'.")

    proposer = ProposerAgent(model_name=args.model)
    skeptic = SkepticAgent(llm_model=args.model)
    orch = Orchestrator(proposer, skeptic, max_rounds=2)

    for crit in criteria:
        orch.run_one(criterion=crit, patient_sentences=patient_sentences)


if __name__ == "__main__":
    _run_from_files()
