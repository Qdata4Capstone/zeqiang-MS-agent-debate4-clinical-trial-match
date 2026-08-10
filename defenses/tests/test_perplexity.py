from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch

from rag_defenses.perplexity import PerplexityDetector, PerplexityScorer


def _mock_causal_lm(loss_value: float):
    """Build mock tokenizer/model objects matching AutoTokenizer/AutoModelForCausalLM's call shape."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = "already-set"
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }

    mock_model = MagicMock()
    mock_outputs = MagicMock()
    mock_outputs.loss = torch.tensor(loss_value)
    mock_model.return_value = mock_outputs
    mock_model.to.return_value = mock_model

    return mock_tokenizer, mock_model


def test_perplexity_detector_fits_and_scores_from_causal_lm_loss():
    mock_tokenizer, mock_model = _mock_causal_lm(loss_value=0.0)  # exp(0) == 1.0

    with patch("rag_defenses.perplexity.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("rag_defenses.perplexity.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        det = PerplexityDetector(model_name="fake-model")
        det.fit(["clean text 1", "clean text 2"])

    assert det.clean_scores is not None
    assert list(det.clean_scores) == [1.0, 1.0]
    assert det.two_sided is True


def test_perplexity_scorer_returns_exp_of_loss_per_text():
    mock_tokenizer, mock_model = _mock_causal_lm(loss_value=1.0)  # exp(1) ~= 2.71828

    with patch("rag_defenses.perplexity.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("rag_defenses.perplexity.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        scorer = PerplexityScorer(model_name="fake-model", device="cpu")
        result = scorer.score_texts(["text a", "text b"])

    assert result.shape == (2,)
    assert result[0].item() == torch.tensor(1.0).exp().item()


def test_perplexity_detector_sets_pad_token_when_missing():
    mock_tokenizer, mock_model = _mock_causal_lm(loss_value=0.0)
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<eos>"

    with patch("rag_defenses.perplexity.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("rag_defenses.perplexity.AutoModelForCausalLM.from_pretrained", return_value=mock_model):
        PerplexityDetector(model_name="fake-model")

    assert mock_tokenizer.pad_token == "<eos>"
