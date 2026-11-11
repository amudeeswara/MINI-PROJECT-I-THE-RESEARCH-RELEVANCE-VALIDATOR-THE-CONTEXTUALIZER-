# 🔎 The Contextualizer — Research Relevance Validator

[![CI](https://github.com/YOUR-USERNAME/contextualizer/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR-USERNAME/contextualizer/actions)
![python](https://img.shields.io/badge/python-3.9%2B-blue) ![deps](https://img.shields.io/badge/runtime%20deps-0-brightgreen) ![license](https://img.shields.io/badge/license-MIT-green)

> **Does this paper *really* belong in my research?**
> The Contextualizer scores candidate papers, references or snippets against *your* research context and tells you **why** each one is relevant, only partially relevant, or a false friend.

Keyword search finds papers that share *words* with your topic. The Contextualizer checks whether they share your *context*:

| Problem | How it is handled |
|---|---|
| Same words, different field (e.g. a *power transformer* vs. a *transformer* neural network) | `domain_drift` flag |
| Overlap only on generic terms ("novel approach", "results show") | Academic boilerplate is down-weighted; `superficial_overlap` flag |
| Paper never mentions your core concept | `missing_core_concept` flag + score penalty |
| Does the source actually support what you claim? | Sentence-level **claim support** with quoted evidence |
| Stale sources | `outdated` flag (`--max-age`) |
| Title that has nothing to do with your topic | `title_mismatch` flag |

Every verdict is **explainable**: component scores, matched / missing terms, flags and evidence sentences. No black box, no API key, **zero runtime dependencies**.

## Quick start

```bash
git clone https://github.com/YOUR-USERNAME/contextualizer && cd contextualizer
pip install -e .

contextualizer demo
```

```
#  SCORE  VERDICT     ID      TITLE / FLAGS
1  0.74   RELEVANT    ref-01  Vision transformers for diabetic retinopathy grading from r…
2  0.37   PARTIAL     ref-02  Deep learning for automated screening of diabetic retinopat…
3  0.20   IRRELEVANT  ref-03  Convolutional networks for pneumonia detection in chest X-r…  [superficial_overlap]
4  0.17   IRRELEVANT  ref-04  Vision transformers at scale for image recognition  [superficial_overlap]
5  0.13   IRRELEVANT  ref-05  Early fault detection in power transformer windings …  [domain_drift]
6  0.00   IRRELEVANT  ref-06  Nitrogen uptake efficiency in winter wheat …  [missing_core_concept]
```

## Usage

### CLI

```bash
# candidates from JSON / JSONL / CSV / BibTeX / text blocks
contextualizer validate -C examples/context.txt -d examples/references.bib
contextualizer validate -c "graph neural networks for drug discovery" -d papers.csv -f md -o report.md
contextualizer validate -C ctx.txt -d refs.bib -f html -o report.html

# pull candidates live from OpenAlex (free, no key)
contextualizer validate -c "early detection of diabetic retinopathy" --openalex "retinopathy transformer" --limit 25

# use as a CI/reference-list linter (exit 1 if anything is irrelevant)
contextualizer validate -C ctx.txt -d refs.bib --fail-on-irrelevant --max-age 10
```

Formats: `table` (default), `json`, `md`, `html`, `csv`.

### Web UI + REST API

```bash
contextualizer serve --port 8000     # → http://127.0.0.1:8000
```

```bash
curl -s localhost:8000/api/validate -H 'Content-Type: application/json' -d '{
  "context": "Early detection of diabetic retinopathy using transformers on fundus images",
  "documents": [{"title": "Vision transformers for retinal fundus grading", "abstract": "..."}],
  "options": {"relevant_threshold": 0.55, "partial_threshold": 0.30}
}'
```

### Python

```python
from contextualizer import RelevanceValidator

v = RelevanceValidator("Early detection of diabetic retinopathy with transformers on fundus images")
for r in v.validate([{"title": "...", "abstract": "...", "year": 2021}]):
    print(r.verdict, r.score, r.flags, r.matched_terms)
    for e in r.evidence:
        print("  supports:", e.claim, "<-", e.sentence)
```

### Optional: dense semantic scoring

```bash
pip install -e ".[semantic]"
contextualizer validate -C ctx.txt -d refs.bib --semantic            # all-MiniLM-L6-v2
```

## How scoring works

For each candidate, five lexical components are computed (each 0–1) over a TF-IDF model fitted on the context + candidates:

| Component | Weight | Meaning |
|---|---|---|
| `similarity` | 0.35 | TF-IDF cosine over unigrams + bigrams (stemmed, stop-words removed) |
| `coverage` | 0.25 | Weighted share of the context's terms found in the document |
| `phrases` | 0.15 | Shared two-word concepts (`retinal_fundus`, `diabetic_retinopathy`) |
| `title` | 0.10 | Weighted share of the title's terms that occur in the context |
| `claims` | 0.15 | Each context sentence is matched to the best document sentence |

The weighted sum is then adjusted by contextual checks (`missing_core_concept` ×0.7, `domain_drift` ×0.85, `superficial_overlap` ×0.8) and mapped to a verdict (`relevant ≥ 0.55`, `partially_relevant ≥ 0.30`). All weights and thresholds are configurable (`RelevanceValidator(weights=..., relevant_threshold=...)`). With `--semantic`, a dense-embedding score is blended in (25 %).

> **Honest limits.** The default engine is lexical + statistical: fast, private, reproducible and explainable, but it can't understand paraphrases the way an LLM or embedding model can (use `--semantic` for that), and with very small candidate pools IDF is weak. Treat verdicts as a *triage aid*, not a replacement for reading the paper.

## Project layout

```
contextualizer/
  text.py        tokenizer, stemmer, stop-words, academic-generic list
  scoring.py     TF-IDF model, cosine
  validator.py   RelevanceValidator — the engine
  loaders.py     JSON / JSONL / CSV / BibTeX / text-block loaders
  fetch.py       OpenAlex candidate fetcher
  report.py      table / md / json / csv / html renderers
  server.py      stdlib web UI + REST API
  semantic.py    optional sentence-transformers backend
  cli.py         `contextualizer` command
tests/           26 tests (pytest)
examples/        sample context, candidates, BibTeX
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Roadmap ideas: LLM-judge backend, PDF full-text input, Zotero export, per-section (intro/methods) contexts, citation-intent classification. PRs welcome — see `CONTRIBUTING.md`.

## License

MIT
