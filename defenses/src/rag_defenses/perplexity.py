from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_defenses.common import BaseDetector


class PerplexityDetector(BaseDetector):
    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        max_length: int = 512,
    ):
        super().__init__(two_sided=True, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        self.clean_scores: np.ndarray | None = None

    @torch.no_grad()
    def _perplexity(self, text: str) -> float:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = outputs.loss.item()
        return float(np.exp(loss))

    def fit(self, clean_texts: Sequence[str]) -> None:
        clean_scores = self.score_texts(clean_texts)
        self.clean_scores = clean_scores
        self.fit_thresholds_from_scores(clean_scores)

    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        scores = [self._perplexity(t) for t in texts]
        return np.array(scores, dtype=np.float64)


class PerplexityScorer:
    def __init__(self, model_name: str = "gpt2", device: str = "cuda"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def score_texts(self, texts):
        scores = []
        for text in texts:
            tokenized = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            scores.append(torch.exp(outputs.loss).detach().cpu())
        return torch.stack(scores).float()
