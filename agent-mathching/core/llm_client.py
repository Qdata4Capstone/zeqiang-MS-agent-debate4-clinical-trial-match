import re
import json
import requests
from typing import Optional, Dict, Any


class OllamaClient:
    """
    Thin HTTP client for local Ollama server.
    Default model: qwen2.5:7b-instruct
    """
    def __init__(self, model: str = "qwen2.5:7b-instruct"):
        self.model = model

    def generate_json(self, prompt: str, temperature: float = 0.0, timeout: int = 120) -> Optional[Dict[str, Any]]:
        """
        Send prompt to Ollama and try to parse a strict-JSON response.
        Falls back to greedy { ... } extraction if the model wrapped JSON in text.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": temperature,
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
            print(f"[LLM][WARN] call failed: {e}")
            return None
