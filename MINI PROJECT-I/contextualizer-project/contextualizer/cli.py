"""Command-line interface: ``contextualizer validate|demo|serve``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .loaders import load_documents, split_context_and_docs
from .report import to_csv, to_html, to_json, to_markdown, to_table
from .validator import RelevanceValidator

DEMO = Path(__file__).parent / "data" / "demo.json"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="contextualizer",
        description="The Contextualizer — validate whether papers/references are relevant to your research context.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("validate", help="score candidate documents against a context")
    v.add_argument("--context", "-c", help="research context as text")
    v.add_argument("--context-file", "-C", help="file containing the research context")
    v.add_argument("--docs", "-d", help="candidates: .json .jsonl .csv .bib or .txt (blocks split by ---)")
    v.add_argument("--openalex", metavar="QUERY", help="fetch candidates from OpenAlex instead of --docs")
    v.add_argument("--limit", type=int, default=20, help="max OpenAlex results (default 20)")
    v.add_argument("--format", "-f", choices=["table", "json", "md", "html", "csv"], default="table")
    v.add_argument("--out", "-o", help="write the report to a file")
    v.add_argument("--relevant", type=float, default=0.55, help="threshold for 'relevant' (default 0.55)")
    v.add_argument("--partial", type=float, default=0.30, help="threshold for 'partially relevant' (default 0.30)")
    v.add_argument("--min-score", type=float, default=0.0, help="hide results below this score")
    v.add_argument("--max-age", type=int, help="flag documents older than N years")
    v.add_argument("--semantic", nargs="?", const="all-MiniLM-L6-v2", metavar="MODEL",
                   help="blend in sentence-transformer embeddings (needs the [semantic] extra)")
    v.add_argument("--fail-on-irrelevant", action="store_true",
                   help="exit with status 1 if any document is irrelevant (handy in CI)")

    sub.add_parser("demo", help="run the bundled example")

    s = sub.add_parser("serve", help="start the local web UI + REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    return p


def _render(fmt: str, results, context: str) -> str:
    return {
        "table": lambda: to_table(results),
        "json": lambda: to_json(results, context),
        "md": lambda: to_markdown(results, context),
        "html": lambda: to_html(results, context),
        "csv": lambda: to_csv(results),
    }[fmt]()


def _validate(args) -> int:
    context = args.context or ""
    if args.context_file:
        context = Path(args.context_file).read_text(encoding="utf-8")
    docs: list = []
    if args.openalex:
        from .fetch import openalex_search

        docs = openalex_search(args.openalex, args.limit)
        context = context or args.openalex
    elif args.docs:
        file_ctx, docs = load_documents(args.docs)
        context = context or file_ctx
    else:
        print("error: provide --docs FILE or --openalex QUERY", file=sys.stderr)
        return 2
    if not context.strip():
        print("error: provide a research context with --context / --context-file", file=sys.stderr)
        return 2

    embedder = None
    if args.semantic:
        from .semantic import load_sentence_transformer

        embedder = load_sentence_transformer(args.semantic)
    validator = RelevanceValidator(
        context, relevant_threshold=args.relevant, partial_threshold=args.partial,
        max_age_years=args.max_age, embedder=embedder,
    )
    results = [r for r in validator.validate(docs) if r.score >= args.min_score]
    text = _render(args.format, results, context)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Report written to {args.out}", file=sys.stderr)
    else:
        print(text)
    if args.fail_on_irrelevant and any(r.verdict == "irrelevant" for r in results):
        return 1
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == "demo":
            ctx, docs = split_context_and_docs(json.loads(DEMO.read_text(encoding="utf-8")))
            print(f"CONTEXT: {ctx}\n")
            print(to_table(RelevanceValidator(ctx).validate(docs)))
            return 0
        if args.cmd == "serve":
            from .server import serve

            serve(args.host, args.port)
            return 0
        return _validate(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
