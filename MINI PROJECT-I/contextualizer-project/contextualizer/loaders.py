"""Load candidate documents from JSON, JSONL, CSV, BibTeX or plain-text blocks."""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from .models import Document

_BLOCK_SPLIT = re.compile(r"^\s*---+\s*$", re.M)


def parse_text_blocks(text: str) -> List[Document]:
    """Blocks separated by a line of ``---``; first line = title, the rest = abstract."""
    docs = []
    for block in _BLOCK_SPLIT.split(text):
        lines = [l for l in block.strip().splitlines()]
        if not lines:
            continue
        title, abstract = lines[0].strip(), " ".join(l.strip() for l in lines[1:])
        docs.append(Document(title=title, abstract=abstract))
    return docs


def _bib_fields(body: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    i, n = 0, len(body)
    while i < n:
        m = re.compile(r"\s*,?\s*([A-Za-z_\-]+)\s*=\s*").match(body, i)
        if not m:
            i += 1
            continue
        key, i = m.group(1).lower(), m.end()
        if i >= n:
            break
        if body[i] == "{":
            depth, j = 1, i + 1
            while j < n and depth:
                depth += (body[j] == "{") - (body[j] == "}")
                j += 1
            value, i = body[i + 1 : j - 1], j
        elif body[i] == '"':
            j = body.find('"', i + 1)
            j = n if j < 0 else j
            value, i = body[i + 1 : j], j + 1
        else:
            m2 = re.compile(r"[^,\n]+").match(body, i)
            value, i = (m2.group(0).strip() if m2 else ""), (m2.end() if m2 else i + 1)
        fields[key] = re.sub(r"[{}]", "", re.sub(r"\s+", " ", value)).strip()
    return fields


def parse_bibtex(text: str) -> List[Document]:
    docs = []
    for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", text):
        if m.group(1).lower() in {"comment", "string", "preamble"}:
            continue
        depth, j = 1, m.end()
        while j < len(text) and depth:
            depth += (text[j] == "{") - (text[j] == "}")
            j += 1
        f = _bib_fields(text[m.end() : j - 1])
        d = Document.from_dict(
            {
                "id": m.group(2),
                "title": f.get("title", ""),
                "abstract": f.get("abstract", ""),
                "year": f.get("year", ""),
                "authors": [a.strip() for a in f.get("author", "").split(" and ") if a.strip()],
                "keywords": f.get("keywords", ""),
                "doi": f.get("doi", ""),
                "url": f.get("url", ""),
            }
        )
        docs.append(d)
    return docs


def split_context_and_docs(payload: Any) -> Tuple[str, List[Document]]:
    """Accept ``{"context":..., "documents":[...]}`` or a bare list."""
    if isinstance(payload, dict):
        docs = payload.get("documents") or payload.get("references") or payload.get("papers") or []
        return str(payload.get("context", "")), [Document.from_dict(d) for d in docs]
    return "", [Document.from_dict(d) for d in payload]


def load_documents(path: Union[str, Path]) -> Tuple[str, List[Document]]:
    """Return ``(context_if_present, documents)``."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    ext = p.suffix.lower()
    if ext == ".json":
        return split_context_and_docs(json.loads(text))
    if ext == ".jsonl":
        return "", [Document.from_dict(json.loads(l)) for l in text.splitlines() if l.strip()]
    if ext in {".csv", ".tsv"}:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t" if ext == ".tsv" else ",")
        return "", [Document.from_dict(r) for r in reader]
    if ext in {".bib", ".bibtex"}:
        return "", parse_bibtex(text)
    return "", parse_text_blocks(text)
