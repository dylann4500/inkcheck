import synth
from inkcheck.load import load_document
from inkcheck.provenance import assess_provenance, classify_producer


def test_a_scribe_like_export_is_a_rendered_raster_with_the_signature(tmp_path):
    page = synth.generated_page(shape=(3168, 2448))
    doc = load_document(synth.image_pdf(tmp_path / "s.pdf", [synth.png_bytes(page)]))
    result = assess_provenance(doc)
    assert result.data["source"] == "rendered-raster"
    assert result.data["signature"] == "scribe"
    assert result.score >= 0.5


def test_the_signature_needs_the_exact_page_and_image_size(tmp_path):
    page = synth.generated_page(shape=(3000, 2400))
    doc = load_document(synth.image_pdf(tmp_path / "s.pdf", [synth.png_bytes(page)]))
    assert assess_provenance(doc).data["signature"] is None


def test_a_note_app_export_is_vector_ink(tmp_path):
    result = assess_provenance(load_document(synth.vector_pdf(tmp_path / "n.pdf")))
    assert result.data["source"] == "vector-ink"
    assert result.data["recognised_text_chars"] > 0
    assert result.score <= 0.1


def test_a_phone_photo_is_camera_or_scan(tmp_path):
    rgb = synth.human_page(shape=(1600, 1200))
    (tmp_path / "p.jpg").write_bytes(synth.photographed(rgb))
    result = assess_provenance(load_document(tmp_path / "p.jpg"))
    assert result.data["source"] == "camera-or-scan"
    assert result.score <= 0.2


def test_producers_are_named():
    assert classify_producer("iOS Version 18.7.8 (Build 22H352) Quartz PDFContext")[0] == "apple-device"
    assert classify_producer("pdf-lib (https://github.com/Hopding/pdf-lib)")[0] == "library"
    assert classify_producer("Adobe Scan for Android 23.1")[0] == "scanner-app"
    assert classify_producer("") is None
