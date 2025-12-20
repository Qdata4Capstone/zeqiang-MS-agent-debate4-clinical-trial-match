

import re, json, os
import requests, re, json
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, model: str = "qwen2.5:7b-instruct"):
        self.model = model

    def generate_json(self, prompt: str, temperature: float = 0.0, timeout: int = 120):
     try:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",        
            "options": {
                "temperature": 0.0,  
                "top_p": 1.0,
                "top_k": 1,             
                "seed": 7,              
                "mirostat": 0,          
                "num_ctx": 8192        
            }
        }
        r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=timeout)
        r.raise_for_status()
        txt = r.json().get("response","")
        try:
            return json.loads(txt)
        except Exception:
            m = re.search(r"\{.*\}", txt, re.S)
            return json.loads(m.group(0)) if m else None
     except Exception as e:
        print(f"[WARN] Ollama call failed: {e}")
        return None


class ProposerAgent:
    def __init__(self, model_name: str = "qwen2.5:7b-instruct"):
        self.llm = OllamaClient(model_name)

    def _evidence_pool_block(self, sentences: List[str]) -> str:
        return os.linesep.join([f"[{i}] {s}" for i, s in enumerate(sentences)])

    def decide(
        self,
        criterion: Dict[str, Any],
        patient_sentences: List[str]
    ) -> Dict[str, Any]:
        ctype = criterion["type"]
        label_policy = ('Choose exactly one: "included" | "not included" | "not enough information".'
                        if ctype == "inclusion" else
                        'Choose exactly one: "exclude" | "not exclude" | "not enough information".')

        prompt_template = """
You are the **Proposer** agent in a clinical trial matching system.

Task
Decide ONE criterion using ONLY the patient evidence pool below. Do NOT use outside knowledge or infer without evidence.
From the FULL pool, select a minimal set of **relevant** sentences by sentence index.



Reasoning mode (CoT—internal)
- Think step-by-step to: (a) identify all the focus concepts; (b) scan the full pool and filter irrelevant sentences;
  (c) derive all the extractions from relevant sentences; (d) test support vs. contradiction (including numeric/temporal/semantic and others);
  (e) check conflicts;(f) before choosing a label, scan the cited sentences for tokens/phrases that imply the opposite of the criterion, if such contra-indicators are present and tied to the focus concept, treat them as **contradiction** evidence; (g) choose the label per the rubric.

Relevance of evidence
- A sentence is **relevant** only if it mentions the focus concept (or a clear synonym) OR provides a numeric/temporal value
  explicitly tied to the concept, OR gives an explicit statement about that concept (affirmation/negation).
- Irrelevant sentences MUST NOT be used as evidence.
  If no relevant sentences exist, evidence MUST be empty and the label MUST be "not enough information".

Output format
Return a SINGLE JSON object with EXACTLY these keys (no others):
- "clause_id": string
- "criterion_type": "inclusion" | "exclusion"
- "reasoning_trace": [               // for auditing; each item: { "step": string, "detail": string }
    ...
  ]
- "claim": string                    // restatement of the criterion
- "evidence": [                      // minimal set; each: {"sent_id": int} (must be relevant)
    ...
  ]
- "extractions": object              // facts derived ONLY from the chosen quotes and tied to the focus concept
                                     // examples:
                                     // contradiction_type: "numeric" | "temporal" | "semantic" | "negation"
- "missing_information": [string]    // REQUIRED IF AND ONLY IF label == "not enough information"; list each missing patient fact/criterion element as a short noun phrase (no full sentences)
- "rationale": string                // verifiable (1–3 sentences), based on your reasoning and evidence
- "label": string                    // <<LABEL_POLICY>>


Decision rubric (absence ≠ negation)
1) Build "extractions" ONLY from the relevant quotes you cite.
2) Label rules (must be satisfied by the quotes/extractions you provide):
   • Inclusion:
     - "included": requires ≥1 **supporting** relevant sentence (or valid threshold comparison) showing the criterion holds.
     - "not included": requires ≥1 **contradicting** relevant sentence, which may be:
         (a) explicit negation ("no X", "denies X", "without X", "negative for X"), OR
         (b) **numeric/temporal/semantic contradiction** (e.g., criterion "BMI > 21" but sentence shows "BMI 20").
     - If neither supporting nor contradicting relevant sentences exist → "not enough information".

   • Exclusion (the label answers: “Should this patient be excluded from the trial because this clause holds?”):
     - "exclude": use ONLY if there is ≥1 supporting, relevant sentence showing that the exclusion condition DOES apply
       to this patient. In this case, the patient should be excluded from the trial.
     - "not exclude": use ONLY if there is ≥1 contradicting, relevant sentence showing that the exclusion condition
       does NOT apply to this patient. This contradiction may be:
         (a) explicit negation ("no history of X", "denies X", "has never had X", "does not have X"), OR
         (b) numeric/temporal/semantic contradiction (e.g., exclusion "BMI > 21" but the sentence shows "BMI 20").
     - If the evidence is neither clearly supporting nor clearly contradicting the exclusion condition → choose
       "not enough information".
       
3) Evidence & extractions requirement
   - If you output "included", "not included", "exclude", or "not exclude":
     • "evidence" MUST contain ≥1 relevant sentence, AND
     • "extractions" MUST contain ≥1 fact about the focus concept that supports/contradicts the criterion.
   - If you cannot provide such relevant evidence/extractions, you MUST output "not enough information".
   - When you output "not enough information", populate "missing_information" with the specific clause elements or patient facts that are missing, expressed as concise noun phrases (e.g., "specific IgE value", "pregnancy status").

4) Conflict handling
   - If both supporting and contradicting relevant sentences are present and cannot be reconciled:
     output "not enough information" and explain briefly in "rationale".

5) Consistency & scope
   - "rationale" must rest only on the sentences/extractions you provided.
   - Do NOT invent content. Use ONLY the evidence pool below.

[Criterion]
id: <<CLAUSE_ID>>
type: <<TYPE>>
text: <<TEXT>>
structured_threshold: <<STRUCTURED>>

[Patient Evidence Pool]
<<POOL>>
""".strip("\n")

        structured = json.dumps({
            "operator": criterion.get("operator"),
            "value": criterion.get("value"),
            "unit": criterion.get("unit"),
        })
        pool_block = self._evidence_pool_block(patient_sentences)

        prompt = (
            prompt_template
            .replace("<<LABEL_POLICY>>", label_policy)
            .replace("<<CLAUSE_ID>>", str(criterion["clause_id"]))
            .replace("<<TYPE>>", str(criterion["type"]))
            .replace("<<TEXT>>", str(criterion["text"]))
            .replace("<<STRUCTURED>>", structured)
            .replace("<<POOL>>", pool_block)
        )

        out = self.llm.generate_json(prompt, temperature=0.2)

        def _keep_in_pool(ev):
            res = []
            if isinstance(ev, list):
                for e in ev:
                    if isinstance(e, dict) and isinstance(e.get("sent_id"), int):
                       sid = e["sent_id"]
                       if 0 <= sid < len(patient_sentences):
                        sent = patient_sentences[sid]
                        res.append({
                        "sent_id": sid,
                        "sent_text": sent,
                        "quote": sent 
                    })
            return res

        def _normalize_missing_info(val):
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

           

        if not out:
            out = {
                "clause_id": criterion["clause_id"],
                "criterion_type": criterion["type"],
                "claim": criterion["text"],
                "evidence": [],
                "extractions": {},
                "missing_information": [
                    "relevant patient evidence"
                ],
                "rationale": "No relevant evidence could be produced; per rubric this implies 'not enough information'.",
                "label": "not enough information",
            }
        else:
            if ctype == "inclusion":
                allowed = {"included","not included","not enough information"}
            else:
                allowed = {"exclude","not exclude","not enough information"}
            if out.get("label") not in allowed:
                out["label"] = "not enough information"

            out["evidence"] = _keep_in_pool(out.get("evidence", []))
            out["clause_id"] = str(out.get("clause_id", criterion["clause_id"]))
            out["criterion_type"] = str(out.get("criterion_type", criterion["type"]))
            out["claim"] = str(out.get("claim", criterion["text"]))
            out["extractions"] = out.get("extractions", {}) or {}
            out["rationale"] = str(out.get("rationale", "")) or ""
            out["missing_information"] = _normalize_missing_info(out.get("missing_information", []))
            if out["label"] == "not enough information":
                if not out["missing_information"]:
                    out["missing_information"] = [
                        "criterion-specific patient data"
                    ]
            else:
                out["missing_information"] = []

        print("\n--- Proposer Output (Decision Record) ---")
        print(json.dumps(out, indent=2))
        return out
    def revise(
        self,
        criterion: Dict[str, Any],
        patient_sentences: List[str],
        prev_decision: Dict[str, Any],
        skeptic_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        ctype = criterion["type"]
        label_policy = ('Choose exactly one: "included" | "not included" | "not enough information".'
                        if ctype == "inclusion" else
                        'Choose exactly one: "exclude" | "not exclude" | "not enough information".')

        template = """
You are the **Proposer** reviser.

Goal
Revise the prior decision using ONLY the patient sentences and the Skeptic feedback. Apply the **minimal changes** needed to resolve cited issues/warnings. Output JSON only.

Output keys (exact):
- "clause_id": string
- "criterion_type": "inclusion" | "exclusion"
- "claim": string
- "evidence": [ { "sent_id": int } ]          // cite 1–2 most relevant sentences; ONLY sent_id
- "extractions": {                             // facts from cited sentences
    "support_facts"?: [string],
    "contra_indicators"?: [string],
    "numbers"?: { ... }                         // e.g., onset_days: 2, latest_episode_end_hours: 0.5
  }
- "rationale": string                           // 1–3 sentences grounded ONLY in the cited sentences
- "label": string                               // <<LABEL_POLICY>>
- "confidence": number                          // 0..1
- "missing_information": [string]               // REQUIRED iff label == "not enough information"; short noun phrases only

Revision rules (follow strictly):
1) Use ONLY the given sentences; no outside knowledge.
2) Fix issues from the Skeptic report **by code** (apply those that matter here):
   - missing_citation: if label is non-NEI, add valid evidence; else change to "not enough information".
   - invalid_sent_id: replace with valid sent_id or drop invalid item.
   - irrelevant_evidence: drop it; re-evaluate label.
   - insufficient_temporal_support: add the needed temporal/numeric sentences or change to NEI.
   - missing_clause_elements: add sentences that cover the uncovered elements or change to NEI.
   - unextracted_relevant_fact: include those facts verbatim into extractions (support_facts/contra_indicators/numbers).
   - direction_mismatch: adjust evidence/extractions and flip label if warranted by citations.
   - possible_world_knowledge: remove any claims not present in the cited sentences/criterion.
   - conflict_evidence: if unresolved, set label to NEI and explain briefly in rationale.
3) Contra-indicators (very important): if cited sentences imply the opposite polarity (e.g., ER, new/first time, hours/days), record them under extractions.contra_indicators and treat as CONTRADICTION.
4) Evidence rule: if label is NOT NEI → evidence must be non-empty and extractions must contain ≥1 supporting/contradicting fact.
5) Determinism & brevity: if multiple sentences fit, pick the LOWEST sent_id; keep evidence to 1–2 sentences.
6) If you choose "not enough information", specify which clause elements or patient facts are missing in "missing_information" as concise noun phrases (else leave it empty).

[Criterion]
id: <<CID>>
type: <<CTYPE>>
text: <<CTEXT>>
structured_threshold: <<STRUCTURED>>

[Patient Sentences]
<<POOL>>

[Prior Decision]
<<PRIOR_DECISION>>

[Skeptic Feedback]
<<SKEPTIC_REPORT>>

Return ONLY the corrected JSON object.
""".strip("\n")

        structured = json.dumps({
            "operator": criterion.get("operator"),
            "value": criterion.get("value"),
            "unit": criterion.get("unit"),
        }, ensure_ascii=False)

        pool_block = "\n".join([f"[{i}] {s}" for i, s in enumerate(patient_sentences)])
        prior_json = json.dumps(prev_decision, ensure_ascii=False)
        skeptic_json = json.dumps(skeptic_report, ensure_ascii=False)

        prompt = (template
                  .replace("<<LABEL_POLICY>>", label_policy)
                  .replace("<<CID>>", str(criterion["clause_id"]))
                  .replace("<<CTYPE>>", str(criterion["type"]))
                  .replace("<<CTEXT>>", str(criterion["text"]))
                  .replace("<<STRUCTURED>>", structured)
                  .replace("<<POOL>>", pool_block)
                  .replace("<<PRIOR_DECISION>>", prior_json)
                  .replace("<<SKEPTIC_REPORT>>", skeptic_json)
                  )

        out = self.llm.generate_json(prompt, temperature=0.0)


        def normalize_evidence(ev):
            res = []
            if isinstance(ev, list):
                for e in ev:
                    sid = e.get("sent_id")
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

        if not out:
            out = {
                "clause_id": prev_decision.get("clause_id", criterion["clause_id"]),
                "criterion_type": prev_decision.get("criterion_type", criterion["type"]),
                "claim": prev_decision.get("claim", criterion["text"]),
                "evidence": prev_decision.get("evidence", []),
                "extractions": prev_decision.get("extractions", {}) or {},
                "rationale": prev_decision.get("rationale", "") or "Revision failed; keep prior decision.",
                "label": prev_decision.get("label", "not enough information"),
                "confidence": prev_decision.get("confidence", 0.6),
                "missing_information": prev_decision.get("missing_information", []),
            }
        else:
            if ctype == "inclusion":
                allowed = {"included","not included","not enough information"}
            else:
                allowed = {"exclude","not exclude","not enough information"}
            if out.get("label") not in allowed:
                out["label"] = "not enough information"

            out["evidence"] = normalize_evidence(out.get("evidence", []))
            out["clause_id"] = str(out.get("clause_id", criterion["clause_id"]))
            out["criterion_type"] = str(out.get("criterion_type", criterion["type"]))
            out["claim"] = str(out.get("claim", criterion["text"]))
            out["extractions"] = out.get("extractions", {}) or {}
            out["rationale"] = str(out.get("rationale", "")) or ""
            try:
                out["confidence"] = float(out.get("confidence", 0.6))
            except:
                out["confidence"] = 0.6
            out["missing_information"] = normalize_missing_info(out.get("missing_information", []))

        if out["label"] == "not enough information":
            if not out["missing_information"]:
                out["missing_information"] = [
                    "criterion-specific patient data"
                ]
        else:
            out["missing_information"] = []

        print("\n--- Proposer Revised Decision ---")
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return out
