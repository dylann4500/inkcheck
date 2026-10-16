"""`inkcheck serve`: a page on this computer to drop files on and read the evidence.

Listens on 127.0.0.1 only. Uploads are read from memory (a temporary file
for the PDF reader, deleted at once) and reports are kept in memory for this
run only; nothing is written anywhere else. Standard library only.
"""

from __future__ import annotations

import json
import secrets
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

from .load import load_bytes
from .report import VERDICT_TEXT, render_html
from .scan import scan_document

MAX_BYTES = 60 * 1024 * 1024
REPORTS: dict[str, str] = {}
LOCK = threading.Lock()

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>inkcheck</title><style>
:root{--bg:#fbfaf7;--card:#fff;--ink:#1d1f24;--muted:#5d6370;--line:#e6e2da;--gen:#c2272d;--sus:#b45309;--ok:#2f6f4f;--none:#6b7280;--accent:#2f5bd3}
@media (prefers-color-scheme:dark){:root{--bg:#15161a;--card:#1d1f24;--ink:#ecebe7;--muted:#a2a6af;--line:#30333a;--gen:#f2545b;--sus:#f59e0b;--ok:#5fbf8a;--none:#9ca3af;--accent:#7ea2ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif}
main{max-width:760px;margin:0 auto;padding:40px 16px 64px}h1{font-size:26px;margin:0}p{margin:6px 0}.muted{color:var(--muted)}
#drop{display:block;margin:22px 0;border:2px dashed var(--line);border-radius:14px;padding:34px 16px;text-align:center;background:var(--card);cursor:pointer}
#drop.over{border-color:var(--accent)}#drop strong{display:block;font-size:17px;margin-bottom:4px}input[type=file]{display:none}
.result{background:var(--card);border:1px solid var(--line);border-left:6px solid var(--none);border-radius:12px;padding:14px 16px;margin:10px 0}
.result.generated{border-left-color:var(--gen)}.result.suspicious{border-left-color:var(--sus)}.result.no-evidence{border-left-color:var(--ok)}
.badge{display:inline-block;font-weight:650;font-size:12px;letter-spacing:.02em;text-transform:uppercase;padding:2px 9px;border-radius:999px;color:#fff;background:var(--none);margin-right:8px}
.badge.generated{background:var(--gen)}.badge.suspicious{background:var(--sus)}.badge.no-evidence{background:var(--ok)}
.name{font-weight:600;word-break:break-all}a{color:var(--accent)}.small{font-size:13px}
</style></head><body><main>
<h1>inkcheck</h1><p class="muted">Finds handwriting that software assembled from samples, and shows the evidence. Everything stays on this computer.</p>
<label id="drop"><input id="files" type="file" multiple accept="application/pdf,image/*"><strong>Drop homework PDFs or photos here</strong><span class="muted">or click to choose files</span></label>
<div id="results"></div>
<p class="muted small">A verdict is evidence for a person to check, never proof about a student. inkcheck never says a page was written by hand; it says when it found no sign of generated handwriting.</p>
</main><script>
const drop = document.getElementById('drop'), input = document.getElementById('files'), results = document.getElementById('results');
const TEXT = %VERDICTS%;
function esc(s){return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
async function check(file){
  const card = document.createElement('div'); card.className = 'result';
  card.innerHTML = `<div class="name">${esc(file.name)}</div><div class="muted small">Checking… (a few seconds a page)</div>`;
  results.prepend(card);
  try {
    const res = await fetch('/scan', {method: 'POST', headers: {'X-Filename': encodeURIComponent(file.name)}, body: file});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    card.className = 'result ' + data.label;
    const reasons = (data.reasons || []).map(r => `<li>${esc(r)}</li>`).join('');
    card.innerHTML = `<div><span class="badge ${data.label}">${esc(TEXT[data.label] || data.label)}</span><span class="name">${esc(file.name)}</span></div>
      <p>${esc(data.headline)}</p>${reasons ? `<ul class="small">${reasons}</ul>` : ''}
      <a href="${data.report}" target="_blank" rel="noopener">Open the evidence report</a> <span class="muted small">· ${data.pages} page${data.pages === 1 ? '' : 's'}, ${data.seconds}s</span>`;
  } catch (e) {
    card.innerHTML = `<div class="name">${esc(file.name)}</div><p>Could not check this file: ${esc(e.message)}</p>`;
  }
}
async function checkAll(list){ for (const f of list) await check(f); }
input.addEventListener('change', () => { checkAll([...input.files]); input.value = ''; });
['dragenter','dragover'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(t => drop.addEventListener(t, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => checkAll([...e.dataTransfer.files]));
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "inkcheck"
    workers: int | None = None

    def log_message(self, fmt, *args):  # keep the terminal quiet; nothing about uploads is logged
        return

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: dict) -> None:
        self._send(status, json.dumps(data).encode(), "application/json")

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("%VERDICTS%", json.dumps(VERDICT_TEXT))
            return self._send(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
        if self.path.startswith("/report/"):
            with LOCK:
                html = REPORTS.get(self.path.rsplit("/", 1)[-1])
            if html:
                return self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")
        self._send(HTTPStatus.NOT_FOUND, b"Not found", "text/plain")

    def do_POST(self):
        if self.path != "/scan":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BYTES:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": f"Files up to {MAX_BYTES // 2**20} MB."})
        data = self.rfile.read(length)
        name = unquote(self.headers.get("X-Filename") or "upload")
        import time

        started = time.time()
        result = scan_document(load_bytes(data, name), workers=self.workers)
        token = secrets.token_urlsafe(12)
        with LOCK:
            REPORTS[token] = render_html(result)
        v = result.verdict
        self._json(HTTPStatus.OK, {"label": v.label, "headline": v.headline, "reasons": v.reasons, "report": f"/report/{token}",
                                   "pages": len(result.document.pages), "seconds": round(time.time() - started, 1)})


def make_server(port: int = 8765, workers: int | None = None) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"workers": workers})
    return ThreadingHTTPServer(("127.0.0.1", port), handler)


def serve(port: int = 8765, workers: int | None = None) -> None:
    server = make_server(port, workers)
    print(f"inkcheck is at http://127.0.0.1:{server.server_address[1]}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
