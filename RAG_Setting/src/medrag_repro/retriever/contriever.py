from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _normalize_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class ContrieverEncoder:
    def __init__(self, model_name: str = "facebook/contriever", device: Optional[str] = None, batch_size: int = 8):
        self.model_name = model_name
        self.device = device or "cpu"
        self.batch_size = batch_size

        torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: Sequence[str], normalize: bool = True) -> np.ndarray:
        all_vecs: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])

            toks = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            toks = {k: v.to(self.device) for k, v in toks.items()}

            outputs = self.model(**toks)
            last_hidden = outputs.last_hidden_state
            mask = toks["attention_mask"].unsqueeze(-1)

            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            vecs = pooled.detach().cpu().numpy().astype("float32")

            if normalize:
                vecs = _normalize_l2(vecs)

            all_vecs.append(vecs)

        if not all_vecs:
            return np.zeros((0, 768), dtype="float32")

        return np.vstack(all_vecs)