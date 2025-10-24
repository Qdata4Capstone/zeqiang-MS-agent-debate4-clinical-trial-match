import os
import json
from typing import List, Dict, Any
from core.llm_client import OllamaClient
from utils.tools import normalize_evidence,_normalize_evidence


class ProposerAgent:

    def __init__(self, model_name: str = "qwen2.5:7b-instruct"):
        self.llm = OllamaClient(model_name)

    # ---- initial decision ----
    def decide(self, criterion: Dict[str, Any], patient_sentences: List[str]) -> Dict[str, Any]:
        """
        Inputs:
          - criterion
          - patient_sentences
        Output:
          - strict JSON decision record
        """
        ctype = criterion["type"]
        label_policy = ('Choose exactly one: "included" | "not included" | "not enough information".'
                        if ctype == "inclusion" else
                        'Choose exactly one: "exclude" | "not exclude" | "not enough information".')

        prompt_template = """
You are the **Proposer** agent in a clinical trial matching system.

Task
Decide ONE criterion using ONLY the patient evidence pool below. Do NOT use outside knowledge or infer without evidence.
From the FULL pool, select a minimal set of **relevant** sentences by sentence index.

Reasoning mode 
- Steps: (a) find all focus concepts; (b) scan full pool, filter irrelevant; (c) derive extractions from relevant sentences;
  (d) test support vs contradiction (numeric/temporal/semantic/negation); (e) check conflicts;
  (f) scan your cited sentences for contra-indicators tied to the concept; (g) choose the label per rubric.

Relevance
- A sentence is relevant only if it mentions the focus concept (or a clear synonym) OR provides numeric/temporal facts tied to it
  OR gives an explicit statement (affirmation/negation). If no relevant sentences exist → evidence MUST be empty and label MUST be "not enough information".

Output format (single JSON):
- "clause_id": string
- "criterion_type": "inclusion" | "exclusion"
- "reasoning_trace": [ { "step": string, "detail": string }, ... ]
- "claim": string
- "evidence": [ {"sent_id": int} ]         // cite by sent_id only
- "extractions": object                    // facts derived ONLY from your evidence; include contradiction_type if any
- "rationale": string                      // 1–3 sentences grounded ONLY in evidence
- "label": string                          // <<LABEL_POLICY>>

Decision rubric 
Inclusion:
  • "included": ≥1 supporting relevant sentence (or valid threshold).
  • "not included": ≥1 contradicting relevant sentence (explicit negation OR numeric/temporal/semantic contradiction).
  • else "not enough information".
Exclusion (analogous):
  • "exclude": ≥1 supporting relevant sentence.
  • "not exclude": ≥1 contradicting relevant sentence.
  • else "not enough information".

[Criterion]
id: <<CLAUSE_ID>>
type: <<TYPE>>
text: <<TEXT>>


[Patient Evidence Pool]
<<POOL>>
""".strip("\n")

        prompt = (
            prompt_template
            .replace("<<LABEL_POLICY>>", label_policy)
            .replace("<<CLAUSE_ID>>", str(criterion["clause_id"]))
            .replace("<<TYPE>>", str(criterion["type"]))
            .replace("<<TEXT>>", str(criterion["text"]))
            .replace("<<POOL>>", self._evidence_pool_block(patient_sentences))
        )

        out = self.llm.generate_json(prompt, temperature=0.0)
        if ctype == "inclusion":
                allowed = {"included", "not included", "not enough information"}
        else:
                allowed = {"exclude", "not exclude", "not enough information"}
        if out.get("label") not in allowed:
                out["label"] = "not enough information"

        out["evidence"] = _normalize_evidence(out.get("evidence", []), patient_sentences)
        out["clause_id"] = str(out.get("clause_id", criterion["clause_id"]))
        out["criterion_type"] = str(out.get("criterion_type", criterion["type"]))
        out["claim"] = str(out.get("claim", criterion["text"]))
        out["extractions"] = out.get("extractions", {}) or {}
        out["rationale"] = str(out.get("rationale", "")) or ""

        print("\n--- Proposer Output (Decision Record) ---")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return out




    # ---- Revision with Skeptic feedback ----
    def revise(
        self,
        criterion: Dict[str, Any],
        patient_sentences: List[str],
        prev_decision: Dict[str, Any],
        skeptic_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctype = criterion["type"]
        label_policy = ('"included" | "not included" | "not enough information"'
                        if ctype == "inclusion" else
                        '"exclude" | "not exclude" | "not enough information"')

        pool_block = "\n".join([f"[{i}] {s}" for i, s in enumerate(patient_sentences)])
        prior_json = json.dumps(prev_decision, ensure_ascii=False)
        skeptic_json = json.dumps(skeptic_report, ensure_ascii=False)

        template = """
You are the **Proposer** reviser.

Goal
Revise the prior decision using ONLY the patient sentences and the Skeptic feedback.Output JSON only.

Output (single JSON)
- revision_trace: [
    {"step":"finding","fix":"what you changed","evidence_delta":[<int>],"note":"brief"}
  ]
- new_decision: {
    "clause_id": "<<CID>>",
    "criterion_type": "<<CTYPE>>",
    "claim": "<<CTEXT>>",
    "evidence": [{"sent_id": int}],
    "extractions": {},
    "rationale": "string",
    "label": <<LABEL_POLICY>>,
  }

You should do:
1) Think about skeptic's feedback and determine whether to change your previous decision or improve your rationale/extractions/evidence based on it.
2) If you change your label, ensure it aligns with the evidence and extractions you provide.
3) If you improve your rationale/extractions/evidence, ensure they address skeptic's concerns.
4) No outside knowledge.

[Criterion]
id: <<CID>>
type: <<CTYPE>>
text: <<CTEXT>>

[Patient Sentences]
<<POOL>>

[Prior Decision]
<<PRIOR_DECISION>>

[Skeptic Feedback]
<<SKEPTIC_REPORT>>

Return ONLY the JSON object above.
""".strip("\n")



        prompt = (template
                  .replace("<<LABEL_POLICY>>", label_policy)
                  .replace("<<CID>>", str(criterion["clause_id"]))
                  .replace("<<CTYPE>>", str(criterion["type"]))
                  .replace("<<CTEXT>>", str(criterion["text"]))
                  .replace("<<POOL>>", pool_block)
                  .replace("<<PRIOR_DECISION>>", prior_json)
                  .replace("<<SKEPTIC_REPORT>>", skeptic_json))

        out = self.llm.generate_json(prompt, temperature=0.0)


        newd = out.get("new_decision", {})
        if ctype == "inclusion":
            allowed = {"included", "not included", "not enough information"}
        else:
            allowed = {"exclude", "not exclude", "not enough information"}
        if newd.get("label") not in allowed:
            newd["label"] = "not enough information"

        newd["evidence"] = _normalize_evidence(newd.get("evidence", []), patient_sentences)
        newd["clause_id"] = str(newd.get("clause_id", criterion["clause_id"]))
        newd["criterion_type"] = str(newd.get("criterion_type", criterion["type"]))
        newd["claim"] = str(newd.get("claim", criterion["text"]))
        newd["extractions"] = newd.get("extractions", {}) or {}
        newd["rationale"] = str(newd.get("rationale", "")) or ""

        print("\n--- Proposer Revision Trace ---")
        print(json.dumps(out.get("revision_trace", []), indent=2, ensure_ascii=False))

        print("\n--- Proposer Revised Decision ---")
        print(json.dumps(newd, indent=2, ensure_ascii=False))
        return newd
