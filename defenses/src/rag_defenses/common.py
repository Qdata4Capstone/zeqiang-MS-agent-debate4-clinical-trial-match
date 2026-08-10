from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Sequence

import numpy as np


class BaseDetector(ABC):
    def __init__(self, two_sided: bool = False, upper_quantile: float = 0.99, lower_quantile: float = 0.01):
        self.two_sided = two_sided
        self.upper_quantile = upper_quantile
        self.lower_quantile = lower_quantile
        self.lower_threshold: float | None = None
        self.upper_threshold: float | None = None

    @abstractmethod
    def fit(self, clean_texts: Sequence[str]) -> None:
        ...

    @abstractmethod
    def score_texts(self, texts: Sequence[str]) -> np.ndarray:
        ...

    def detect(self, texts: Sequence[str]) -> list[bool]:
        scores = self.score_texts(texts)
        if self.two_sided:
            assert self.lower_threshold is not None and self.upper_threshold is not None
            return ((scores < self.lower_threshold) | (scores > self.upper_threshold)).tolist()
        else:
            assert self.upper_threshold is not None
            return (scores > self.upper_threshold).tolist()

    def fit_thresholds_from_scores(self, clean_scores: np.ndarray) -> None:
        if self.two_sided:
            self.lower_threshold = float(np.quantile(clean_scores, self.lower_quantile))
            self.upper_threshold = float(np.quantile(clean_scores, self.upper_quantile))
        else:
            self.upper_threshold = float(np.quantile(clean_scores, self.upper_quantile))
