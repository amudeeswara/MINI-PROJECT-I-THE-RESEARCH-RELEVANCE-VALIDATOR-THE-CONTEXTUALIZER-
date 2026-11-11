"""Optional: pull candidate papers from OpenAlex (free, no API key)."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from .models import Document

API = "https://api.openalex.org/works"


def rebuild_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    if not inverted:
        return ""
    positions = {}
    for word, idxs in inverted.items():
        for i in idxs:
            positions[i] = word
    return " ".join(positions[i] for i in sorted(positions))


def parse_openalex(payload: Dict[str, Any]) -> List[Document]:
    docs = []
    for w in payload.get("results", []):
        docs.append(
            Document(
                id=str(w.get("id", "")).rsplit("/", 1)[-1],
                title=w.get("title") or "",
                abstract=rebuild_abstract(w.get("abstract_inverted_index")),
                year=w.get("publication_year"),
                authors=[
                    (a.get("author") or {}).get("display_name", "")
                    for a in (w.get("authorships") or [])
                ][:6],
                doi=w.get("doi") or "",
                url=w.get("id") or "",
            )
        )
    return docs


def openalex_search(query: str, limit: int = 20, mailto: Optional[str] = None, timeout: int = 20) -> List[Document]:
    params = {
        "search": query,
        "per-page": str(max(1, min(limit, 100))),
        "select": "id,doi,title,publication_year,abstract_inverted_index,authorships",
    }
    if mailto:
        params["mailto"] = mailto
    req = urllib.request.Request(
        f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "contextualizer/0.1"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
        return parse_openalex(json.load(resp))
