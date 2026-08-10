from __future__ import annotations

import numpy as np
import pytest

from rag_defenses.common import BaseDetector


class _ConstantScoreDetector(BaseDetector):
    """Minimal concrete subclass for testing BaseDetector's shared logic."""

    def __init__(self, scores_by_text, **kwargs):
        super().__init__(**kwargs)
        self.scores_by_text = scores_by_text

    def fit(self, clean_texts):
        pass

    def score_texts(self, texts):
        return np.array([self.scores_by_text[t] for t in texts], dtype=np.float64)


def test_fit_thresholds_from_scores_one_sided_sets_only_upper():
    det = _ConstantScoreDetector({}, two_sided=False, upper_quantile=0.9)
    det.fit_thresholds_from_scores(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert det.lower_threshold is None
    assert det.upper_threshold == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9))


def test_fit_thresholds_from_scores_two_sided_sets_both():
    det = _ConstantScoreDetector({}, two_sided=True, lower_quantile=0.1, upper_quantile=0.9)
    det.fit_thresholds_from_scores(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert det.lower_threshold == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.1))
    assert det.upper_threshold == pytest.approx(np.quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.9))


def test_detect_one_sided_flags_scores_above_upper_threshold():
    det = _ConstantScoreDetector({"clean": 1.0, "poison": 100.0}, two_sided=False, upper_quantile=0.99)
    det.upper_threshold = 10.0

    assert det.detect(["clean", "poison"]) == [False, True]


def test_detect_two_sided_flags_scores_outside_either_threshold():
    det = _ConstantScoreDetector({"low": -50.0, "mid": 1.0, "high": 50.0}, two_sided=True)
    det.lower_threshold = -10.0
    det.upper_threshold = 10.0

    assert det.detect(["low", "mid", "high"]) == [True, False, True]
