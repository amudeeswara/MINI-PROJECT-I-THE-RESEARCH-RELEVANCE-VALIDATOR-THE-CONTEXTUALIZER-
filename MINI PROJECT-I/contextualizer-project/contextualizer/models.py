"""Data classes shared across the package."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional

_ABSTRACT_KEYS = ("abstract", "summary", "text", "body", "description", "snippet")
_TITLE_KEYS = ("title", "name", "heading")
_YEAR_KEYS = ("year", "publication_year", "published", "date")


def _first(d: Mapping[str, Any], keys) -> Any:
    lowered = {str(k).lower(): v for k, v in d.items()}
    for k in keys:
        if lowered.get(k) not in (None, ""):
            return lowered[k]
    return None


def _to_year(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    s = str(value)
    for i in range(len(s) - 3):
        chunk = s[i : i + 4]
        if chunk.isdigit() and 1500 <= int(chunk) <= 2200:
            return int(chunk)
    return None


def _to_list(value: Any) -> List[str]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    sep = ";" if ";" in str(value) else ","
    return [v.strip() for v in str(value).split(sep) if v.strip()]


@dataclass
class Document:
    """A candidate paper / reference / snippet to be checked against a research context."""

    title: str = ""
    abstract: str = ""
    id: str = ""
    year: Optional[int] = None
    authors: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    doi: str = ""
    url: str = ""

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Document":
        return cls(
            title=str(_first(d, _TITLE_KEYS) or ""),
            abstract=str(_first(d, _ABSTRACT_KEYS) or ""),
            id=str(_first(d, ("id", "key", "citekey")) or ""),
            year=_to_year(_first(d, _YEAR_KEYS)),
            authors=_to_list(_first(d, ("authors", "author"))),
            keywords=_to_list(_first(d, ("keywords", "keyword", "tags"))),
            doi=str(_first(d, ("doi",)) or ""),
            url=str(_first(d, ("url", "link")) or ""),
        )

    @property
    def full_text(self) -> str:
        parts = [self.title, self.abstract, " ".join(self.keywords)]
        return ". ".join(p for p in parts if p)


@dataclass
class Evidence:
    claim: str
    sentence: str
    score: float


@dataclass
class ValidationResult:
    id: str
    title: str
    score: float
    verdict: str  # relevant | partially_relevant | irrelevant
    components: Dict[str, float]
    matched_terms: List[str]
    missing_terms: List[str]
    evidence: List[Evidence]
    flags: List[str]
    reasons: List[str]
    year: Optional[int] = None
    doi: str = ""
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
