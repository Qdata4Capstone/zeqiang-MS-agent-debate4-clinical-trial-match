from __future__ import annotations

import os
from openai import OpenAI


def load_openai_client() -> OpenAI:
    base_url = os.getenv("OPENAI_BASE_URL")
    api_key = os.getenv("OPENAI_API_KEY")

    # Ollama local mode: default to a placeholder API key.
    if base_url and "localhost:11434" in base_url and not api_key:
        api_key = "ollama"

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    return OpenAI(api_key=api_key, base_url=base_url)


def chat_completion(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""
