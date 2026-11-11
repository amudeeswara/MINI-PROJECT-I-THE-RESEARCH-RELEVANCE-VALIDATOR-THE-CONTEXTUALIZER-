"""Tiny dependency-free web UI + REST API (``contextualizer serve``)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

from . import __version__
from .loaders import parse_text_blocks
from .models import Document
from .report import summary_counts
from .validator import RelevanceValidator

MAX_BODY = 5 * 1024 * 1024


def run_validation(payload: Dict[str, Any]) -> Dict[str, Any]:
    context = str(payload.get("context", "")).strip()
    if not context:
        raise ValueError("'context' is required")
    raw = payload.get("documents")
    if isinstance(raw, str):
        docs = parse_text_blocks(raw)
    elif isinstance(raw, list):
        docs = [Document.from_dict(d) if isinstance(d, dict) else Document(abstract=str(d)) for d in raw]
    else:
        raise ValueError("'documents' must be a list or a text string")
    if not docs:
        raise ValueError("no documents supplied")
    opts = payload.get("options") or {}
    validator = RelevanceValidator(
        context,
        relevant_threshold=float(opts.get("relevant_threshold", 0.55)),
        partial_threshold=float(opts.get("partial_threshold", 0.30)),
        max_age_years=opts.get("max_age_years"),
    )
    results = validator.validate(docs)
    return {"summary": summary_counts(results), "results": [r.to_dict() for r in results]}


class Handler(BaseHTTPRequestHandler):
    server_version = f"Contextualizer/{__version__}"

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode())

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._send(200, UI.encode(), "text/html")
        elif self.path == "/api/health":
            self._json(200, {"status": "ok", "version": __version__})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/validate":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_BODY:
                return self._json(413, {"error": "payload too large"})
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._json(200, run_validation(payload))
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})

    def log_message(self, fmt, *args):  # quiet
        pass


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    srv = make_server(host, port)
    print(f"Contextualizer running at http://{host}:{srv.server_address[1]}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


UI = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>The Contextualizer</title>
<style>
:root{--bg:#fff;--fg:#1f2328;--mut:#57606a;--bd:#d0d7de;--ac:#0969da;--ok:#1a7f37;--mid:#9a6700;--bad:#cf222e}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#8d96a0;--bd:#30363d;--ac:#4493f8;--ok:#3fb950;--mid:#d29922;--bad:#f85149}}
body{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:980px;margin:0 auto;padding:1.5rem}
h1{margin:0}p.s{color:var(--mut);margin-top:.2rem}
textarea{width:100%;box-sizing:border-box;background:transparent;color:inherit;border:1px solid var(--bd);border-radius:8px;padding:.6rem;font:inherit}
button{background:var(--ac);color:#fff;border:0;border-radius:8px;padding:.6rem 1.2rem;font:inherit;cursor:pointer;margin-top:.6rem}
label{font-weight:600;display:block;margin:1rem 0 .3rem}
.card{border:1px solid var(--bd);border-radius:10px;padding:1rem;margin:.8rem 0}
.v{font-weight:700}.relevant{color:var(--ok)}.partially_relevant{color:var(--mid)}.irrelevant{color:var(--bad)}
.bar{display:grid;grid-template-columns:80px 1fr 38px;gap:8px;align-items:center;font-size:12px;color:var(--mut)}
.bar i{display:block;height:7px;background:var(--ac);border-radius:4px}
.flag{background:#fff8c5;color:#4d2d00;border-radius:4px;padding:0 6px;margin-left:6px;font-size:12px}
.m{color:var(--mut);font-size:13px}#sum{font-weight:600;margin-top:1rem}
</style>
<h1>🔎 The Contextualizer</h1>
<p class="s">Research Relevance Validator — does each source really belong in your research context?</p>
<label for="ctx">Research context (question, abstract or paragraph)</label>
<textarea id="ctx" rows="5"></textarea>
<label for="docs">Candidate sources — separate with a line containing <code>---</code> (first line = title, rest = abstract)</label>
<textarea id="docs" rows="10"></textarea>
<button id="go">Validate</button> <button id="demo" style="background:transparent;color:var(--ac);border:1px solid var(--bd)">Load demo</button>
<div id="sum"></div><div id="out"></div>
<script>
const $=id=>document.getElementById(id), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
$('demo').onclick=()=>{
 $('ctx').value="Early detection of diabetic retinopathy using transformer models on retinal fundus images. We investigate whether vision transformers outperform convolutional networks at detecting early-stage lesions such as microaneurysms.";
 $('docs').value=["Vision transformers for diabetic retinopathy grading from retinal fundus photographs\\nWe fine-tune vision transformers to grade diabetic retinopathy on fundus photographs and detect microaneurysms.","Early fault detection in power transformer windings\\nPower transformer failures are costly. We present early detection of winding faults using dissolved gas analysis.","Nitrogen uptake in winter wheat\\nField trials measured soil nitrogen and crop yield."].join("\\n---\\n");
};
$('go').onclick=async()=>{
 $('out').innerHTML='';$('sum').textContent='Scoring…';
 try{
  const r=await fetch('/api/validate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({context:$('ctx').value,documents:$('docs').value})});
  const d=await r.json(); if(!r.ok) throw new Error(d.error);
  const s=d.summary; $('sum').textContent=`${s.relevant} relevant · ${s.partially_relevant} partial · ${s.irrelevant} irrelevant`;
  $('out').innerHTML=d.results.map((x,i)=>`<div class="card"><b>${i+1}. ${esc(x.title)}</b><br>
   <span class="v ${x.verdict}">${x.verdict.replace('_',' ').toUpperCase()}</span> · score ${x.score.toFixed(2)}
   ${x.flags.map(f=>`<span class="flag">${esc(f)}</span>`).join('')}
   ${Object.entries(x.components).map(([k,v])=>`<div class="bar"><span>${k}</span><div><i style="width:${v*100}%"></i></div><span>${v.toFixed(2)}</span></div>`).join('')}
   <div class="m">Matched: ${esc(x.matched_terms.join(', ')||'—')}</div>
   <ul class="m">${x.reasons.map(t=>`<li>${esc(t)}</li>`).join('')}</ul></div>`).join('');
 }catch(e){$('sum').textContent='Error: '+e.message}
};
</script></html>"""
