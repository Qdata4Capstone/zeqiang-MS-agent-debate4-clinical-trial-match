import json
import urllib.error
import urllib.request


class OllamaError(RuntimeError):
    """Raised when an Ollama request fails."""


def generate_json(
    model,
    prompt,
    system=None,
    base_url="http://localhost:11434",
    temperature=0.7,
    timeout=300,
):
    """Call Ollama and return a parsed JSON object from the response text."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }
    if system:
        payload["system"] = system

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaError(
            f"Failed to connect to Ollama at {base_url}: {exc}"
        ) from exc

    text = body.get("response", "").strip()
    if not text:
        raise OllamaError("Ollama returned an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise OllamaError(
            "Ollama response was not valid JSON. "
            f"Raw response: {text[:500]}"
        ) from exc
