"""The core engine: score how relevant candidate documents are to a research context."""
from __future__ import annotations

import datetime as _dt
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Union

from .models import Document, Evidence, ValidationResult
from .scoring import TfidfModel, clamp, cosine, dense_cosine
from .text import bigrams, features, split_sentences, stem, tokenize

DEFAULT_WEIGHTS: Dict[str, float] = {
    "similarity": 0.35,  # TF-IDF cosine over unigrams + bigrams
    "coverage": 0.25,    # share of the context's (weighted) key terms the document contains
    "phrases": 0.15,     # shared two-word concepts, e.g. "retinal_fundus"
    "title": 0.10,       # does the document's title talk about the same things as the context?
    "claims": 0.15,      # does some sentence of the document echo each sentence of the context?
}

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
DocLike = Union[Document, Mapping[str, Any], str]


def _coerce(doc: DocLike, idx: int) -> Document:
    if isinstance(doc, Document):
        d = doc
    elif isinstance(doc, str):
        d = Document(abstract=doc)
    else:
        d = Document.from_dict(doc)
    if not d.id:
        d.id = f"doc-{idx + 1}"
    return d


class RelevanceValidator:
    """Validate candidate documents against a research *context*.

    Parameters
    ----------
    context:
        Your research question / abstract / paragraph being written (string, dict or Document).
    weights:
        Override any of :data:`DEFAULT_WEIGHTS`.
    relevant_threshold, partial_threshold:
        Score cut-offs for the verdicts ``relevant`` and ``partially_relevant``.
    core_terms:
        How many of the context's most informative terms form its "core concept" set.
    max_age_years:
        If set, documents older than this get an ``outdated`` flag.
    embedder:
        Optional function ``list[str] -> list[vector]``. When given, dense semantic similarity is
        blended in with ``semantic_weight`` (see :mod:`contextualizer.semantic`).
    """

    def __init__(
        self,
        context: DocLike,
        *,
        weights: Optional[Mapping[str, float]] = None,
        relevant_threshold: float = 0.55,
        partial_threshold: float = 0.30,
        core_terms: int = 6,
        max_age_years: Optional[int] = None,
        embedder: Optional[Embedder] = None,
        semantic_weight: float = 0.25,
        now_year: Optional[int] = None,
    ):
        if partial_threshold > relevant_threshold:
            raise ValueError("partial_threshold must be <= relevant_threshold")
        self.context_doc = _coerce(context, 0)
        self.context_text = self.context_doc.full_text
        if not tokenize(self.context_text):
            raise ValueError("The research context is empty (no usable words).")
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            unknown = set(weights) - set(w)
            if unknown:
                raise ValueError(f"Unknown weight(s): {sorted(unknown)}")
            w.update(weights)
        total = sum(w.values()) or 1.0
        self.weights = {k: v / total for k, v in w.items()}
        self.relevant_threshold = relevant_threshold
        self.partial_threshold = partial_threshold
        self.core_terms = core_terms
        self.max_age_years = max_age_years
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.now_year = now_year or _dt.date.today().year

    # ------------------------------------------------------------------ public API
    def validate(self, docs: Sequence[DocLike]) -> List[ValidationResult]:
        """Score every document; results are sorted best-first."""
        documents = [_coerce(d, i) for i, d in enumerate(docs)]
        if not documents:
            return []

        ctx_tokens = tokenize(self.context_text)
        doc_tokens = [tokenize(d.full_text) for d in documents]
        model = TfidfModel([features(ctx_tokens)] + [features(t) for t in doc_tokens])

        ctx_vec = model.vectorize(features(ctx_tokens))
        ctx_weights = model.term_weights(ctx_tokens)  # unigram weights
        ctx_total = sum(ctx_weights.values()) or 1.0
        core = [t for t, _ in sorted(ctx_weights.items(), key=lambda kv: -kv[1])[: self.core_terms]]
        ctx_bi = set(bigrams(ctx_tokens))
        claims = [
            (s, model.vectorize(features(tokenize(s))))
            for s in split_sentences(self.context_text)
            if len(tokenize(s)) >= 3
        ] or [(self.context_text, ctx_vec)]

        self._surface = {}
        for raw in tokenize(self.context_text, stem_words=False):
            self._surface.setdefault(stem(raw), raw)

        semantic = self._semantic_scores(documents)

        results = [
            self._score_one(
                doc, toks, model, ctx_vec, ctx_weights, ctx_total, core, ctx_bi, claims,
                semantic[i] if semantic else None,
            )
            for i, (doc, toks) in enumerate(zip(documents, doc_tokens))
        ]
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def validate_one(self, doc: DocLike) -> ValidationResult:
        return self.validate([doc])[0]

    # ------------------------------------------------------------------ internals
    def _semantic_scores(self, documents: List[Document]) -> Optional[List[float]]:
        if not self.embedder:
            return None
        vecs = self.embedder([self.context_text] + [d.full_text for d in documents])
        ctx = vecs[0]
        return [clamp((dense_cosine(ctx, v) - 0.2) / 0.6) for v in vecs[1:]]

    def _verdict(self, score: float) -> str:
        if score >= self.relevant_threshold:
            return "relevant"
        if score >= self.partial_threshold:
            return "partially_relevant"
        return "irrelevant"

    def _score_one(
        self, doc, toks, model, ctx_vec, ctx_weights, ctx_total, core, ctx_bi, claims, sem
    ) -> ValidationResult:
        doc_set = set(toks)
        feats = features(toks)

        # 1. similarity
        sim = clamp(cosine(ctx_vec, model.vectorize(feats)) / 0.5)

        # 2. weighted coverage of context terms
        matched = [t for t in ctx_weights if t in doc_set]
        coverage = sum(ctx_weights[t] for t in matched) / ctx_total

        # 3. phrase overlap
        shared_bi = ctx_bi & set(bigrams(toks))
        phrases = clamp(len(shared_bi) / max(1, min(len(ctx_bi), 8)))

        # 4. title agreement
        title_toks = set(tokenize(doc.title))
        if title_toks:
            tw = model.term_weights(list(title_toks))
            title = sum(w for t, w in tw.items() if t in ctx_weights) / (sum(tw.values()) or 1.0)
        else:
            title = 0.0

        # 5. claim support (+ evidence)
        sentences = split_sentences(". ".join(p for p in (doc.title, doc.abstract) if p)) or [doc.title]
        sent_vecs = [
            (s, model.vectorize(features(tokenize(s)))) for s in sentences if tokenize(s)
        ]
        claim_scores: List[float] = []
        evidence: List[Evidence] = []
        for claim, cvec in claims:
            best_s, best = "", 0.0
            for s, svec in sent_vecs:
                c = cosine(cvec, svec)
                if c > best:
                    best_s, best = s, c
            claim_scores.append(clamp(best / 0.5))
            if best >= 0.12:
                evidence.append(Evidence(claim=claim, sentence=best_s, score=round(best, 3)))
        claims_score = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0
        evidence.sort(key=lambda e: -e.score)

        components = {
            "similarity": sim,
            "coverage": coverage,
            "phrases": phrases,
            "title": title,
            "claims": claims_score,
        }
        lexical = sum(self.weights[k] * v for k, v in components.items())
        if sem is not None:
            components["semantic"] = sem
            score = (1 - self.semantic_weight) * lexical + self.semantic_weight * sem
        else:
            score = lexical

        # contextual checks ---------------------------------------------------
        flags: List[str] = []
        reasons: List[str] = []
        core_hits = [t for t in core if t in doc_set]
        core_ratio = len(core_hits) / max(1, len(core))
        dw = model.term_weights(toks)
        precision = sum(w for t, w in dw.items() if t in ctx_weights) / (sum(dw.values()) or 1.0)
        if not core_hits:
            flags.append("missing_core_concept")
            score *= 0.7
            reasons.append("None of the context's core concepts appear in this document.")
        elif core_ratio < 0.34 and coverage < 0.25 and len(matched) >= 2:
            flags.append("superficial_overlap")
            score *= 0.8
            reasons.append(
                f"Overlap is superficial: only {len(core_hits)}/{len(core)} core concepts are shared "
                "and little of the context is covered."
            )
        if len(matched) >= 2 and precision < 0.25:
            flags.append("domain_drift")
            score *= 0.85
            reasons.append(
                f"Only {precision:.0%} of this document's own vocabulary relates to your context: "
                "it shares some words but is mostly about something else (possible different sense or field)."
            )
        if self.max_age_years and doc.year and self.now_year - doc.year > self.max_age_years:
            flags.append("outdated")
            reasons.append(
                f"Published {doc.year}, older than the {self.max_age_years}-year freshness limit."
            )
        if len(toks) < 8:
            flags.append("insufficient_text")
            reasons.append("Very little text to judge from (add an abstract for a reliable verdict).")
        if title_toks and title == 0.0 and score >= self.partial_threshold:
            flags.append("title_mismatch")
            reasons.append("The document's title shares nothing with your context.")

        score = clamp(score)
        verdict = self._verdict(score)

        reasons.insert(0, f"Weighted coverage of context terms: {coverage:.0%}; "
                          f"shared key phrases: {len(shared_bi)}.")
        if evidence:
            reasons.append(f"Strongest supporting sentence: \"{evidence[0].sentence}\"")

        def pretty(t: str) -> str:
            return self._surface.get(t, t)

        missing = [t for t, _ in sorted(ctx_weights.items(), key=lambda kv: -kv[1])
                   if t not in doc_set][:8]
        matched_sorted = sorted(matched, key=lambda t: -ctx_weights[t])[:12]
        return ValidationResult(
            id=doc.id,
            title=doc.title or (doc.abstract[:60] + "…"),
            score=round(score, 4),
            verdict=verdict,
            components={k: round(v, 4) for k, v in components.items()},
            matched_terms=[pretty(t) for t in matched_sorted],
            missing_terms=[pretty(t) for t in missing],
            evidence=evidence[:3],
            flags=flags,
            reasons=reasons,
            year=doc.year,
            doi=doc.doi,
            url=doc.url,
        )


def validate(context: DocLike, docs: Sequence[DocLike], **kwargs) -> List[ValidationResult]:
    """One-shot convenience wrapper."""
    return RelevanceValidator(context, **kwargs).validate(docs)
