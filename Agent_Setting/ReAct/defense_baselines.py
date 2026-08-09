from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rag_infra.defenses.l2_norm import l2_norm_score as _l2_norm_score_np


@dataclass
class QuantileStats:
    threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float


@dataclass
class PerplexityStats:
    lower_threshold: float
    upper_threshold: float
    clean_scores: torch.Tensor
    false_positive_rate: float


def fit_upper_quantile(clean_scores: torch.Tensor, quantile: float = 0.99) -> QuantileStats:
    threshold = torch.quantile(clean_scores, quantile).item()
    false_positive_rate = (clean_scores > threshold).float().mean().item()
    return QuantileStats(
        threshold=threshold,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
    )


def fit_two_sided_quantile(clean_scores: torch.Tensor, quantile: float = 0.99) -> PerplexityStats:
    tail = max((1 - quantile) / 2, 1e-4)
    lower = torch.quantile(clean_scores, tail).item()
    upper = torch.quantile(clean_scores, 1 - tail).item()
    false_positive_rate = ((clean_scores < lower) | (clean_scores > upper)).float().mean().item()
    return PerplexityStats(
        lower_threshold=lower,
        upper_threshold=upper,
        clean_scores=clean_scores,
        false_positive_rate=false_positive_rate,
    )


def l2_norm_scores(embeddings: torch.Tensor) -> torch.Tensor:
    device = embeddings.device
    scores_np = _l2_norm_score_np(embeddings.detach().cpu().float().numpy())
    return torch.from_numpy(scores_np).float().to(device)


def l2_distance_scores(embeddings: torch.Tensor, clean_reference: torch.Tensor) -> torch.Tensor:
    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    distances = torch.cdist(embeddings.float(), clean_reference.float())
    return distances.min(dim=1).values


def leave_one_out_l2_distance_scores(clean_reference: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(clean_reference.float(), clean_reference.float())
    diagonal_mask = torch.eye(distances.shape[0], device=distances.device, dtype=torch.bool)
    distances.masked_fill_(diagonal_mask, float("inf"))
    return distances.min(dim=1).values


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
