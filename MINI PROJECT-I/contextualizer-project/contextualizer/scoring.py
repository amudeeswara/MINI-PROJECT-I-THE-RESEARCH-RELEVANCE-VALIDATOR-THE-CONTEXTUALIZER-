"""TF-IDF primitives with academic-boilerplate dampening."""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List

from .text import GENERIC_STEMS

GENERIC_WEIGHT = 0.3


class TfidfModel:
    """Smoothed TF-IDF fitted on a (small) corpus of feature lists."""

    def __init__(self, corpus: Iterable[List[str]]):
        self.df: Counter = Counter()
        self.n = 0
        for doc in corpus:
            self.n += 1
            self.df.update(set(doc))

    def idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df.get(term, 0) + 1)) + 1.0

    @staticmethod
    def damp(term: str) -> float:
        parts = term.split("_")
        return GENERIC_WEIGHT if all(p in GENERIC_STEMS for p in parts) else 1.0

    def term_weights(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        return {
            t: (1.0 + math.log(c)) * self.idf(t) * self.damp(t) for t, c in tf.items()
        }

    def vectorize(self, tokens: List[str]) -> Dict[str, float]:
        vec = self.term_weights(tokens)
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}


def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def dense_cosine(a: List[float], b: List[float]) -> float:
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)
