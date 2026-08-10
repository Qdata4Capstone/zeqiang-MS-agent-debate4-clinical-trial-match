from __future__ import annotations

from typing import Dict, Sequence


def options_block(options: Dict[str, str]) -> str:
    return "\n".join([f"{k}. {v}" for k, v in sorted(options.items())])


def answer_with_context_system_prompt() -> str:
    return (
        "You are a medical multiple-choice QA assistant. Use the provided context only. "
        "Return ONLY a JSON object like {\"predicted_option\": \"A\"}."
    )


def answer_with_context_user_prompt(contexts: Sequence[str], question: str, options: Dict[str, str]) -> str:
    ctx = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    return f"""
{ctx}

Question:
{question}

Options:
{options_block(options)}

Return only JSON.
""".strip()
