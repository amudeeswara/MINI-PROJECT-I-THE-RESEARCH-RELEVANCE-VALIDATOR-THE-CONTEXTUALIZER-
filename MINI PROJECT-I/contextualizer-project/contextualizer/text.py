"""Dependency-free text normalisation helpers."""
from __future__ import annotations

import re
from typing import List

STOPWORDS = frozenset(
    """a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from further
    had has have having he her here hers him his how i if in into is it its itself just may me might
    more most must my no nor not of off on once only or other our out over own same she should so some
    such than that the their them then there these they this those through to too under until up us
    very was we were what when where which while who whom why will with within without would you your
    et al""".split()
)

# Words that are common in academic prose but say little about *topic*.
# They are down-weighted (not removed) so that two papers do not look related
# merely because both "propose a novel approach and show significant results".
ACADEMIC_GENERIC = frozenset(
    """study studies paper papers result results method methods methodology approach approaches
    analysis analyses analyze analyse model models data dataset datasets use used using based propose
    proposed present presented show shown shows demonstrate demonstrated novel new research work works
    effect effects impact role system systems framework review overview towards toward via case
    evaluate evaluation evaluated experiment experiments experimental performance improve improved
    improvement application applications technique techniques problem problems task tasks investigate
    investigated investigation report reported significant significantly high low large small
    different various several however thus therefore among two three one first second provide provides
    provided found find finding findings""".split()
)

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def stem(word: str) -> str:
    """A small, deterministic suffix stripper (good enough for matching, not linguistics)."""
    if len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    for suf in (
        "ization", "isation", "ational", "ation", "ition", "ment", "ness",
        "ingly", "edly", "ing", "ed", "ly", "ion",
    ):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            word = word[: -len(suf)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


GENERIC_STEMS = frozenset(stem(w) for w in ACADEMIC_GENERIC)


def tokenize(text: str, *, stem_words: bool = True, drop_stop: bool = True) -> List[str]:
    """Lower-case, split on non-word characters, drop stop-words, optionally stem."""
    out: List[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        if len(raw) < 2:
            continue
        if raw.isdigit() and len(raw) != 4:  # keep years, drop stray numbers
            continue
        if drop_stop and raw in STOPWORDS:
            continue
        out.append(stem(raw) if stem_words else raw)
    return out


def bigrams(tokens: List[str]) -> List[str]:
    return [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


def features(tokens: List[str]) -> List[str]:
    """Unigrams + adjacent-content-word bigrams."""
    return tokens + bigrams(tokens)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]
