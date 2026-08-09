import os
import re

import requests


DEFAULT_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b-instruct")
DEFAULT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "300"))


def _tokenize_for_compatibility(text):
    return re.findall(r"\n| [A-Za-z]+|[A-Za-z]+| ?\d+| ?\[[^\]]*|\]| ?[:.,!?;()_-]", text)


def ollama_generate(
    prompt,
    system_prompt=None,
    stop=None,
    temperature=0.0,
    max_tokens=256,
    model=None,
    timeout=DEFAULT_TIMEOUT,
):
    payload = {
        "model": model or DEFAULT_OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }

    if system_prompt:
        payload["system"] = system_prompt
    if stop:
        payload["options"]["stop"] = stop

    response = requests.post(
        f"{DEFAULT_OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"]


def ollama_completion(
    prompt,
    stop=None,
    return_probs=False,
    system_prompt=None,
    temperature=0.0,
    max_tokens=256,
    model=None,
    timeout=DEFAULT_TIMEOUT,
):
    text = ollama_generate(
        prompt=prompt,
        system_prompt=system_prompt,
        stop=stop,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        timeout=timeout,
    )

    if not return_probs:
        return text

    tokens = _tokenize_for_compatibility(text)
    if not tokens:
        tokens = [text]

    return {
        "text": text,
        "logprobs": {
            "tokens": tokens,
            "token_logprobs": [0.0 for _ in tokens],
        },
    }
