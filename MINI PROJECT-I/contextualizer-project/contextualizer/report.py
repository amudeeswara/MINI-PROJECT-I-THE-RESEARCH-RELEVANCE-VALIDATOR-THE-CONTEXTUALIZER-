"""Render validation results as table / markdown / JSON / CSV / HTML."""
from __future__ import annotations

import csv
import html
import io
import json
from typing import List, Sequence

from .models import ValidationResult

ICON = {"relevant": "✅", "partially_relevant": "🟡", "irrelevant": "❌"}
LABEL = {"relevant": "RELEVANT", "partially_relevant": "PARTIAL", "irrelevant": "IRRELEVANT"}


def summary_counts(results: Sequence[ValidationResult]) -> dict:
    out = {"relevant": 0, "partially_relevant": 0, "irrelevant": 0}
    for r in results:
        out[r.verdict] += 1
    return out


def to_json(results: Sequence[ValidationResult], context: str = "") -> str:
    return json.dumps(
        {"context": context, "summary": summary_counts(results), "results": [r.to_dict() for r in results]},
        indent=2,
        ensure_ascii=False,
    )


def to_csv(results: Sequence[ValidationResult]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "title", "score", "verdict", "flags", "matched_terms", "year", "doi"])
    for r in results:
        w.writerow([r.id, r.title, r.score, r.verdict, ";".join(r.flags), ";".join(r.matched_terms), r.year or "", r.doi])
    return buf.getvalue()


def to_table(results: Sequence[ValidationResult], width: int = 60) -> str:
    rows = [("#", "SCORE", "VERDICT", "ID", "TITLE / FLAGS")]
    for i, r in enumerate(results, 1):
        title = r.title if len(r.title) <= width else r.title[: width - 1] + "…"
        flags = f"  [{', '.join(r.flags)}]" if r.flags else ""
        rows.append((str(i), f"{r.score:.2f}", LABEL[r.verdict], r.id, title + flags))
    widths = [max(len(row[c]) for row in rows) for c in range(4)]
    lines = []
    for n, row in enumerate(rows):
        lines.append("  ".join(row[c].ljust(widths[c]) for c in range(4)) + "  " + row[4])
        if n == 0:
            lines.append("-" * (sum(widths) + 8 + width))
    c = summary_counts(results)
    lines.append("")
    lines.append(f"{c['relevant']} relevant · {c['partially_relevant']} partial · {c['irrelevant']} irrelevant")
    return "\n".join(lines)


def to_markdown(results: Sequence[ValidationResult], context: str = "") -> str:
    c = summary_counts(results)
    out: List[str] = ["# Research Relevance Report", ""]
    if context:
        out += ["**Context**", "", f"> {context.strip()}", ""]
    out += [
        f"**Summary:** {c['relevant']} relevant · {c['partially_relevant']} partially relevant · "
        f"{c['irrelevant']} irrelevant",
        "",
        "| # | Score | Verdict | Title | Flags |",
        "|---|-------|---------|-------|-------|",
    ]
    for i, r in enumerate(results, 1):
        out.append(
            f"| {i} | {r.score:.2f} | {ICON[r.verdict]} {LABEL[r.verdict]} | "
            f"{r.title.replace('|', '/')} | {', '.join(r.flags) or '—'} |"
        )
    out.append("")
    out.append("## Details")
    for r in results:
        out += ["", f"### {ICON[r.verdict]} {r.title} (`{r.id}`) — {r.score:.2f}", ""]
        out.append("Components: " + ", ".join(f"{k} {v:.2f}" for k, v in r.components.items()))
        if r.matched_terms:
            out.append(f"\nMatched terms: {', '.join(r.matched_terms)}")
        if r.missing_terms:
            out.append(f"\nMissing context terms: {', '.join(r.missing_terms)}")
        for reason in r.reasons:
            out.append(f"- {reason}")
        for e in r.evidence[:2]:
            out.append(f"- Evidence ({e.score:.2f}): “{e.sentence}” ← supports “{e.claim}”")
    return "\n".join(out) + "\n"


def to_html(results: Sequence[ValidationResult], context: str = "") -> str:
    e = html.escape
    color = {"relevant": "#1a7f37", "partially_relevant": "#9a6700", "irrelevant": "#cf222e"}
    cards = []
    for i, r in enumerate(results, 1):
        bars = "".join(
            f"<div class='bar'><span>{e(k)}</span><i style='width:{v*100:.0f}%'></i><b>{v:.2f}</b></div>"
            for k, v in r.components.items()
        )
        flags = "".join(f"<em>{e(f)}</em>" for f in r.flags)
        ev = "".join(f"<blockquote>{e(x.sentence)}</blockquote>" for x in r.evidence[:2])
        cards.append(
            f"<section><h3>{i}. {e(r.title)}</h3>"
            f"<p><strong style='color:{color[r.verdict]}'>{LABEL[r.verdict]}</strong> · score {r.score:.2f} · "
            f"<code>{e(r.id)}</code> {flags}</p>{bars}"
            f"<p class='m'>Matched: {e(', '.join(r.matched_terms)) or '—'}</p>{ev}"
            f"<ul>{''.join(f'<li>{e(x)}</li>' for x in r.reasons)}</ul></section>"
        )
    return f"""<!doctype html><meta charset="utf-8"><title>Research Relevance Report</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;color:#1f2328}}
section{{border:1px solid #d0d7de;border-radius:8px;padding:1rem;margin:1rem 0}}
.bar{{display:grid;grid-template-columns:90px 1fr 40px;gap:8px;align-items:center;font-size:12px}}
.bar i{{display:block;height:8px;background:#0969da;border-radius:4px}}
em{{background:#fff8c5;border-radius:4px;padding:0 6px;margin-left:4px;font-size:12px}}
blockquote{{margin:.5rem 0;padding-left:.8rem;border-left:3px solid #d0d7de;color:#57606a}}
.m{{color:#57606a;font-size:13px}}</style>
<h1>Research Relevance Report</h1><p><b>Context:</b> {e(context)}</p>{''.join(cards)}"""


RENDERERS = {"json": to_json, "markdown": to_markdown, "md": to_markdown, "html": to_html}
