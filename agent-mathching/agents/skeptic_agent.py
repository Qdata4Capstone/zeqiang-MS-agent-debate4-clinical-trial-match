import json
from typing import List, Dict, Any, Optional
from core.llm_client import OllamaClient


class SkepticAgent:
    """
    Skeptic:
    - Audits the Proposer's decision. Does NOT assign a replacement label.
    - Checks: evidence validity/relevance, world knowledge, evidence↔claim relation,
      missing clause elements, overlooked facts, contradictions/conflicts.
    """

    def __init__(self, llm_model: str = "qwen2.5:7b-instruct"):
        self.llm = OllamaClient(llm_model)
        
    #reduce hallucination by rule
    #def _rule_audit(self, criterion: Dict[str, Any], proposer: Dict[str, Any], sents: List[str]) -> Optional[Dict[str, Any]]:

    def _llm_audit(self, criterion: Dict[str, Any], proposer: Dict[str, Any], sents: List[str]) -> Optional[Dict[str, Any]]:
        if not self.llm:
            return None

        # Normalize evidence
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
# avoid common hallucinations by LLM
        template = """
You are the **Skeptic** auditor. Audit the Proposer's decision USING ONLY the given criterion and the Proposer's cited sentences.

Return STRICT JSON with keys:
- issues (array of objects)    // each: { "code": string, "msg": string, "detail"?: any }
- warnings (array of objects)  // same shape as issues
- support_status (string)      // "support" | "contradict" | "insufficient" | "conflict"
- short_explanation (string)   // sentences grounded ONLY in the cited sentences; 

Codebook:
- "missing_clause_elements": The criterion contains multiple required elements (e.g., threshold + qualifier),
  but the Proposer's extractions/rationale cover only a subset.
- "unextracted_relevant_fact": The cited sent_text contains additional relevant facts (negations, numbers, time bounds, qualifiers)
  that are NOT reflected in Proposer's extractions or rationale.
- "irrelevant_evidence": Cited sentence does not mention the focus concept nor provide numeric/temporal facts tied to it.
- "possible_world_knowledge": Rationale/extractions introduce facts NOT present in the citations nor the criterion.


Audit checklist:
1) Evidence validity & scope: judge ONLY on the cited sent_text.
2) Relevance: must be tied to the focus concept(s) or provide numeric/temporal facts.
3) Relationship to the criterion: decide "support_status": support / contradict / insufficient / conflict.
4) Completeness within citations: if required elements not covered → "missing_clause_elements";
   if cited sentence has overlooked relevant facts → "unextracted_relevant_fact".

[Criterion]
type: <<CTYPE>>
text: <<CTEXT>>

[Proposer Decision]
label: <<PLABEL>>
evidence: <<EVIDENCE>>
extractions: <<EXTRACTIONS>>
rationale: <<RATIONALE>>
reasoning: <<REASONING>>

[Patient Sentences]
<<POOL>>

Rules:
- Use ONLY the Proposer's cited sentences (sent_text). Output JSON only.
""".strip("\n")

        prompt = (template
                  .replace("<<CTYPE>>", str(criterion["type"]))
                  .replace("<<CTEXT>>", str(criterion["text"]))
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

        support_status = (llm_result or {}).get("support_status")
        proposer_label = proposer_output.get("label")

        # Verdict policy
        critical_codes = {"missing_citation", "invalid_sent_id"}
        has_critical = any(
            (f.get("code") in critical_codes) and (proposer_label != "not enough information")
            for f in findings
        )
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
           # "criterion_type": criterion["type"],
           # "proposer_label": proposer_output.get("label"),
           # "skeptic_verdict": verdict,  # accept | needs_revision | block
           # "llm_support_status": support_status,
            "short_explanation": (llm_result or {}).get("short_explanation", "")
        }
