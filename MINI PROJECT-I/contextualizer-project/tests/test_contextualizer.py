import json
import threading
import urllib.request
from pathlib import Path

import pytest

from contextualizer import Document, RelevanceValidator, validate
from contextualizer.cli import main
from contextualizer.fetch import parse_openalex, rebuild_abstract
from contextualizer.loaders import load_documents, parse_bibtex, parse_text_blocks
from contextualizer.report import to_csv, to_html, to_json, to_markdown, to_table
from contextualizer.server import make_server
from contextualizer.text import split_sentences, stem, tokenize

ROOT = Path(__file__).resolve().parent.parent
DEMO = json.loads((ROOT / "contextualizer" / "data" / "demo.json").read_text())


@pytest.fixture(scope="module")
def results():
    return {r.id: r for r in RelevanceValidator(DEMO["context"]).validate(DEMO["documents"])}


# ---------------------------------------------------------------- text
def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The detection of images") == ["detect", "imag"]


def test_stem_is_stable_for_inflections():
    assert stem("detecting") == stem("detection") == stem("detected")


def test_split_sentences():
    assert len(split_sentences("One idea. Another idea! Third?")) == 3


# ---------------------------------------------------------------- scoring behaviour
def test_ranking_order(results):
    order = sorted(results, key=lambda k: -results[k].score)
    assert order[0] == "ref-01" and order[1] == "ref-02" and order[-1] == "ref-06"


def test_verdicts(results):
    assert results["ref-01"].verdict == "relevant"
    assert results["ref-02"].verdict == "partially_relevant"
    for k in ("ref-03", "ref-04", "ref-05", "ref-06"):
        assert results[k].verdict == "irrelevant"


def test_homonym_trap_is_flagged(results):
    # "transformer" the electrical device must not be mistaken for the ML architecture
    assert "domain_drift" in results["ref-05"].flags


def test_unrelated_doc_missing_core(results):
    assert "missing_core_concept" in results["ref-06"].flags and results["ref-06"].score == 0


def test_evidence_and_terms(results):
    r = results["ref-01"]
    assert r.evidence and "fundus" in r.evidence[0].sentence.lower()
    assert "fundus" in r.matched_terms and "retinopathy" in r.matched_terms


def test_scores_bounded(results):
    assert all(0 <= r.score <= 1 for r in results.values())


def test_outdated_flag():
    r = RelevanceValidator(DEMO["context"], max_age_years=5, now_year=2026).validate(DEMO["documents"])
    assert "outdated" in {x.id: x for x in r}["ref-02"].flags


def test_custom_weights_and_validation():
    with pytest.raises(ValueError):
        RelevanceValidator("x y z", weights={"nope": 1})
    with pytest.raises(ValueError):
        RelevanceValidator("   ")
    with pytest.raises(ValueError):
        RelevanceValidator("context words", relevant_threshold=0.2, partial_threshold=0.5)


def test_plain_strings_and_one_shot():
    out = validate("graph neural networks for molecules", ["Graph neural networks predict molecular properties", "Baking sourdough bread at home"])
    assert out[0].score > out[1].score


def test_empty_candidate_list():
    assert RelevanceValidator("deep learning for retina").validate([]) == []


def test_semantic_blend_with_fake_embedder():
    def fake(texts):  # identical vectors => semantic similarity 1.0
        return [[1.0, 0.0] for _ in texts]

    r = RelevanceValidator(DEMO["context"], embedder=fake).validate(DEMO["documents"])
    assert all("semantic" in x.components for x in r)


# ---------------------------------------------------------------- loaders
def test_text_blocks():
    docs = parse_text_blocks("Title A\nabstract a\n---\nTitle B\nabstract b")
    assert [d.title for d in docs] == ["Title A", "Title B"]


def test_bibtex():
    docs = parse_bibtex((ROOT / "examples" / "references.bib").read_text())
    assert len(docs) == 2 and docs[0].id == "dosovitskiy2021" and docs[0].year == 2023
    assert docs[0].authors == ["Doe, Jane", "Roe, Richard"]


def test_load_documents_formats(tmp_path):
    ctx, docs = load_documents(ROOT / "examples" / "candidates_with_context.json")
    assert ctx and len(docs) == 6
    csvp = tmp_path / "d.csv"
    csvp.write_text("title,abstract,year\nA paper,About retinas,2020\n")
    assert load_documents(csvp)[1][0].year == 2020
    jl = tmp_path / "d.jsonl"
    jl.write_text('{"title":"x","abstract":"y"}\n')
    assert load_documents(jl)[1][0].title == "x"


def test_document_aliases():
    d = Document.from_dict({"Name": "T", "summary": "S", "publication_year": 2020, "author": "A; B"})
    assert (d.title, d.abstract, d.year, d.authors) == ("T", "S", 2020, ["A", "B"])


# ---------------------------------------------------------------- openalex parsing (offline)
def test_openalex_parsing():
    inv = {"Hello": [0], "world": [1]}
    assert rebuild_abstract(inv) == "Hello world"
    docs = parse_openalex({"results": [{"id": "https://openalex.org/W1", "title": "T", "publication_year": 2022,
                                         "abstract_inverted_index": inv, "authorships": [{"author": {"display_name": "A"}}]}]})
    assert docs[0].id == "W1" and docs[0].authors == ["A"]


# ---------------------------------------------------------------- reports
def test_reports(results):
    rs = sorted(results.values(), key=lambda r: -r.score)
    assert json.loads(to_json(rs, "ctx"))["summary"]["relevant"] == 1
    assert "ref-01" in to_table(rs) and "| 1 |" in to_markdown(rs, "ctx")
    assert to_csv(rs).startswith("id,title")
    assert "&lt;b&gt;" in to_html(rs, "<b>")  # escaped


# ---------------------------------------------------------------- CLI
def test_cli_validate_json(capsys):
    code = main(["validate", "-C", str(ROOT / "examples/context.txt"), "-d", str(ROOT / "examples/candidates.json"), "-f", "json"])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["results"][0]["id"] == "ref-01"


def test_cli_fail_on_irrelevant():
    args = ["validate", "-C", str(ROOT / "examples/context.txt"), "-d", str(ROOT / "examples/candidates.json"), "-f", "csv"]
    assert main(args + ["--fail-on-irrelevant"]) == 1


def test_cli_errors(capsys):
    assert main(["validate", "-d", str(ROOT / "examples/candidates.json")]) == 2
    assert main(["validate", "-c", "x", "-d", "missing.json"]) == 2


def test_cli_demo(capsys):
    assert main(["demo"]) == 0 and "ref-01" in capsys.readouterr().out


# ---------------------------------------------------------------- server
@pytest.fixture()
def server():
    srv = make_server("127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req)


def test_api_validate(server):
    body = json.load(_post(server + "/api/validate", {"context": DEMO["context"], "documents": DEMO["documents"]}))
    assert body["results"][0]["id"] == "ref-01"


def test_api_text_blocks_and_errors(server):
    body = json.load(_post(server + "/api/validate", {"context": "retinal imaging", "documents": "Retinal imaging study\nfundus photos\n---\nWheat\nsoil"}))
    assert len(body["results"]) == 2
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(server + "/api/validate", {"documents": []})
    assert e.value.code == 400


def test_health_and_ui(server):
    assert json.load(urllib.request.urlopen(server + "/api/health"))["status"] == "ok"
    assert b"Contextualizer" in urllib.request.urlopen(server + "/").read()
