import json

import synth
from inkcheck.cli import main


def test_scan_flags_a_generated_pdf_and_writes_the_evidence(tmp_path, capsys):
    page = synth.generated_page(seed=31, shape=(3168, 2448), n_glyphs=40, size=60, width=4.5)
    pdf = synth.image_pdf(tmp_path / "homework.pdf", [synth.png_bytes(page)])
    code = main(["scan", str(pdf), "--html", str(tmp_path / "r.html"), "--json", str(tmp_path / "r.json"), "--workers", "1"])
    assert code == 2
    assert "GENERATED" in capsys.readouterr().out
    report = (tmp_path / "r.html").read_text()
    assert report.count("data:image/png;base64,") >= 3
    data = json.loads((tmp_path / "r.json").read_text())
    assert data["verdict"]["label"] == "generated"
    assert data["layers"]["provenance"]["data"]["signature"] == "scribe"


def test_scan_finds_nothing_on_a_note_app_export(tmp_path, capsys):
    pdf = synth.vector_pdf(tmp_path / "note.pdf")
    code = main(["scan", str(pdf), "--workers", "1"])
    out = capsys.readouterr().out
    assert code in (0, 3)
    assert "GENERATED" not in out and "SUSPICIOUS" not in out


def test_an_unreadable_file_exits_3(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 nonsense")
    assert main(["scan", str(bad), "--quiet"]) == 3
