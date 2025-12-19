
import re
import json
import requests
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, model: str = "qwen2.5:7b-instruct"):
        self.model = model

    def generate_json(self, prompt: str, temperature: float = 0.0, timeout: int = 120) -> Optional[Dict[str, Any]]:
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
            txt = r.json().get("response", "")
            try:
                return json.loads(txt)
            except Exception:
                m = re.search(r"\{.*\}", txt, re.S)
                return json.loads(m.group(0)) if m else None
        except Exception as e:
            print(f"[Skeptic][WARN] LLM call failed: {e}")
            return None

class SkepticAgent:
    def __init__(self, llm_model: str = "qwen2.5:7b-instruct"):
        self.llm = OllamaClient(llm_model)

    def _llm_audit(self, criterion: Dict[str, Any], proposer: Dict[str, Any], sents: List[str]) -> Optional[Dict[str, Any]]:
        if not self.llm:
            return None
        ev_norm: List[Dict[str, Any]] = []
        for ev in (proposer.get("evidence") or []):
            sid = ev.get("sent_id")
            if isinstance(sid, int) and 0 <= sid < len(sents):
                sent_text = ev.get("sent_text") or sents[sid]
                ev_norm.append({"sent_id": sid, "sent_text": sent_text})
            else:
                ev_norm.append({"sent_id": sid, "sent_text": None})

        pool_block = "\n".join([f"[{i}] {s}" for i, s in enumerate(sents)])
        ev_block = json.dumps(ev_norm, ensure_ascii=False)
        extr = json.dumps(proposer.get("extractions", {}) or {}, ensure_ascii=False)
        proposer_label = proposer.get("label")

        template = """
You are the **Skeptic** auditor. Audit the Proposer's decision USING ONLY the given criterion and the Proposer's cited sentences.

Return STRICT JSON with keys:
- issues (array of objects)    // each: { "code": string, "msg": string, "detail"?: any }
- warnings (array of objects)  // same shape as issues
- support_status (string)      // "support" | "contradict" | "insufficient" | "conflict"
- short_explanation (string)   // 1–2 sentences grounded ONLY in the cited sentences; DO NOT suggest any label.

Assumptions & scope:
- Each evidence item has {"sent_id": int, "sent_text": string|null}. Treat "sent_text" as the canonical full sentence for that sent_id.
- Your analysis MUST rely ONLY on the cited sent_text values. Do NOT use uncited sentences from the pool.

Codebook:
- "missing_clause_elements": The criterion contains multiple required elements (e.g., threshold + qualifier like "primary reason"),
  but the Proposer's extractions/rationale cover only a subset; cite which element(s) are uncovered.
- "unextracted_relevant_fact": The cited sent_text contains additional relevant facts (e.g., negations, numbers, time bounds, qualifiers)
  that are NOT reflected in Proposer's extractions or rationale; list the overlooked fact(s) verbatim from the sent_text.
- "missing_citation": Use ONLY if Proposer's label is NOT "not enough information" AND the evidence array is empty or contains null/invalid sent_text (hard error).
- "invalid_sent_id": Evidence has a sent_id that cannot correspond to any sentence (e.g., negative or beyond pool length).
- "irrelevant_evidence": Cited sentence does not mention the focus concept nor provide numeric/temporal facts tied to it.
- "insufficient_temporal_support": Criterion is temporal/numeric but the cited sentences lack enough temporal/numeric facts to decide.
  If Proposer's label is "not enough information", report this as a WARNING code "insufficient"; otherwise as an ISSUE.
- "possible_world_knowledge": Rationale/extractions introduce NEW clinical facts/entities NOT present in the cited sentences nor in the criterion text.
  Meta statements like "no relevant sentences were found" are NOT world knowledge.
- "conflict_evidence": Both support and contradiction appear and cannot be reconciled.

Audit checklist:
1) Evidence validity & scope: base your judgment ONLY on the cited sent_text; do not import uncited information.
2) Relevance: evidence must be relevant to the focus concept(s) or provide numeric/temporal facts tied to it.
3) Relationship to the criterion: decide overall "support_status": support / contradict / insufficient / conflict.
4) Completeness within citations:
   - For each cited sent_text, verify whether all relevant facts tied to the criterion (negations, numeric/temporal values, qualifiers like "primary reason", time windows) have been reflected in extractions/rationale.
   - If a required clause element is uncovered → "missing_clause_elements" (ISSUE unless the label is NEI with status insufficient, then WARNING).
   - If the cited sentence contains overlooked relevant facts → "unextracted_relevant_fact" (ISSUE unless label is NEI with status insufficient, then WARNING).

[Criterion]
type: <<CTYPE>>
text: <<CTEXT>>
structured_threshold: {"operator": <<OPR>>, "value": <<VAL>>, "unit": <<UNIT>>}

[Proposer Decision]
label: <<PLABEL>>
evidence: <<EVIDENCE>>
extractions: <<EXTRACTIONS>>
rationale: <<RATIONALE>>
reasoning: <<REASONING>>

[Patient Sentences (for reference; do NOT quote unless cited)]
<<POOL>>

Rules:
- Use ONLY the Proposer's cited sentences (sent_text). Output JSON only.
""".strip("\n")

        prompt = (template
                  .replace("<<CTYPE>>", str(criterion["type"]))
                  .replace("<<CTEXT>>", str(criterion["text"]))
                  .replace("<<OPR>>", json.dumps(criterion.get("operator")))
                  .replace("<<VAL>>", json.dumps(criterion.get("value")))
                  .replace("<<UNIT>>", json.dumps(criterion.get("unit")))
                  .replace("<<PLABEL>>", proposer_label or "")
                  .replace("<<EVIDENCE>>", ev_block)
                  .replace("<<EXTRACTIONS>>", extr)
                  .replace("<<RATIONALE>>", str(proposer.get("rationale") or ""))
                  .replace("<<REASONING>>", json.dumps(proposer.get("reasoning") or proposer.get("reasoning_trace") or []))
                  .replace("<<POOL>>", pool_block))

        return self.llm.generate_json(prompt, temperature=0.0)

    def audit(self, criterion: Dict[str, Any], proposer_output: Dict[str, Any], patient_sentences: List[str]) -> Dict[str, Any]:
        llm_result = self._llm_audit(criterion, proposer_output, patient_sentences)

        findings: List[Dict[str, Any]] = []
        if llm_result:
            for it in llm_result.get("issues", []) or []:
                findings.append({"severity": "issue-llm", **it})
            for it in llm_result.get("warnings", []) or []:
                findings.append({"severity": "warning-llm", **it})

        # verdict, based on support_status + proposer label + findings
        support_status = (llm_result or {}).get("support_status")
        proposer_label = proposer_output.get("label")

        critical_codes = {"missing_citation", "invalid_sent_id"}
        has_critical = any(
            (f.get("code") in critical_codes) and (proposer_label != "not enough information")
            for f in findings)
        completeness_issue_codes = {"missing_clause_elements", "unextracted_relevant_fact"}

        has_completeness_issue = any(
          (f.get("severity", "").startswith("issue")) and (f.get("code") in completeness_issue_codes)
          for f in findings
        )

        if has_critical:
            verdict = "block"
        else:
            if support_status == "insufficient" and proposer_label == "not enough information":
                verdict = "accept"
            elif support_status == "support" and proposer_label in {"included", "exclude"}:
                verdict = "accept"
            elif support_status == "contradict" and proposer_label in {"not included", "not exclude"}:
                verdict = "accept"
            elif has_completeness_issue:
                verdict = "needs_revision"
            elif support_status == "conflict":
                verdict = "needs_revision"
            else:
                verdict = "needs_revision"

        return {
           # "clause_id": proposer_output.get("clause_id", criterion["clause_id"]),
            #"criterion_type": criterion["type"],
            #"proposer_label": proposer_output.get("label"),
            #"skeptic_verdict": verdict,  # accept | needs_revision | block
            #"findings": findings,
            #"llm_support_status": support_status,
            "short_explanation": (llm_result or {}).get("short_explanation", "")
        }
