from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class QAItem:
    qid: str
    question: str
    options: Dict[str, str]
    correct_option: str
    correct_text: str
    target_option: Optional[str] = None
    target_text: Optional[str] = None


@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    abstract: str
    text: str
    source: str = "pubmed"


@dataclass
class PoisonDoc:
    poison_id: str
    target_qid: str
    question: str
    target_option: str
    target_text: str
    I_text: str
    full_text: str
    generation_verified: bool
    attempts: int
