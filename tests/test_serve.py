import json
import threading
import urllib.request

import synth
from inkcheck.serve import make_server


def test_the_page_scans_an_upload_and_serves_its_report(tmp_path):
    server = make_server(0, workers=1)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        home = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
        assert "Drop homework PDFs" in home
        page = synth.generated_page(seed=51, shape=(3168, 2448), n_glyphs=40, size=60, width=4.5)
        pdf = synth.image_pdf(tmp_path / "hw.pdf", [synth.png_bytes(page)]).read_bytes()
        request = urllib.request.Request(f"http://127.0.0.1:{port}/scan", data=pdf, method="POST",
                                         headers={"X-Filename": "hw.pdf", "Content-Type": "application/pdf"})
        answer = json.loads(urllib.request.urlopen(request).read())
        assert answer["label"] == "generated"
        report = urllib.request.urlopen(f"http://127.0.0.1:{port}{answer['report']}").read().decode()
        assert "Letters that are copies of one another" in report
    finally:
        server.shutdown()
        server.server_close()
