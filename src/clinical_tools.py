# clinical_tools.py
import re
from typing import List, Dict, Any, Optional, Tuple


def sentence_split(text: str) -> List[str]:
    text = text.strip()
    text = re.sub(r'([.!?])(\s+)', r'\1@@SPLIT@@', text)
    return [s.strip() for s in text.split('@@SPLIT@@') if s.strip()]

def parse_criteria(trial_text: str) -> List[Dict[str, Any]]:
    def infer_threshold(t: str) -> Tuple[Optional[str], Optional[float], Optional[str]]:
        pats = [
            (r"\bless than\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", "<"),
            (r"\bgreater than\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", ">"),
            (r"\bno more than\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", "<="),
            (r"\bat least\s+(\d+(?:\.\d+)?)\s+([A-Za-z]+)\b", ">="),
            (r"\b<=\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\b", "<="),
            (r"\b>=\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\b", ">="),
            (r"\b<\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\b", "<"),
            (r"\b>\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\b", ">"),
            (r"\b==\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\b", "=="),
            (r"\b!=\s*(\d+(?:\.\d+)?)\s*([A-Za-z]+)\b", "!="),
        ]
        for pat, op in pats:
            m = re.search(pat, t, re.I)
            if m:
                try:
                    val = float(m.group(1)); unit = m.group(2).lower()
                    return op, val, unit
                except:
                    pass
        return None, None, None

    lines = [ln.strip() for ln in trial_text.splitlines()]
    mode = None
    inc, exc = 1, 1
    out = []
    for ln in lines:
        if not ln:
            continue
        low = ln.lower()
        if "inclusion criteria" in low:
            mode = "inclusion"; continue
        if "exclusion criteria" in low:
            mode = "exclusion"; continue
        if mode in ("inclusion","exclusion"):
            cid = f"inc_{inc:02d}" if mode=="inclusion" else f"exc_{exc:02d}"
            op, val, unit = infer_threshold(ln)
            out.append({
                "clause_id": cid,
                "type": mode,
                "text": ln,
                "operator": op,
                "value": val,
                "unit": unit
            })
            if mode == "inclusion": inc += 1
            else: exc += 1
    return out

def build_evidence_pool(patient_text: str) -> List[Dict[str, Any]]:
    sents = sentence_split(patient_text)
    return [{"sent_id": i, "text": s} for i, s in enumerate(sents)]
