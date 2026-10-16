import numpy as np

import synth
from inkcheck.load import load_document


def test_image_only_pdf_gives_the_images_own_pixels(tmp_path):
    page = synth.generated_page(shape=(3168, 2448))
    pdf = synth.image_pdf(tmp_path / "scribe.pdf", [synth.png_bytes(page)])
    doc = load_document(pdf)
    assert doc.error is None
    [p] = doc.pages
    assert p.facts.full_page_image
    assert p.pixels.shape == (3168, 2448, 3)
    assert round(p.dpi) == 288
    assert np.array_equal(p.pixels, page)
    assert doc.metadata["producer"].startswith("pdf-lib")


def test_vector_pdf_is_rendered_with_its_ink_counted(tmp_path):
    doc = load_document(synth.vector_pdf(tmp_path / "note.pdf"))
    [p] = doc.pages
    assert not p.facts.full_page_image
    assert p.facts.vector_segments > 100
    assert p.facts.invisible_text_chars > 0 and p.facts.visible_text_chars == 0
    assert p.pixels.shape == (round(792 * 4), round(612 * 4), 3)
    assert p.dpi == 288


def test_png_and_jpeg_files_load(tmp_path):
    rgb = synth.human_page(shape=(400, 300))
    (tmp_path / "page.png").write_bytes(synth.png_bytes(rgb))
    (tmp_path / "photo.jpg").write_bytes(synth.photographed(rgb))
    png = load_document(tmp_path / "page.png")
    jpg = load_document(tmp_path / "photo.jpg")
    assert png.pages[0].pixels.shape == (400, 300, 3)
    assert png.pages[0].facts.image_codecs == ["png"]
    assert jpg.pages[0].facts.image_codecs == ["jpeg"]
    assert not jpg.pages[0].dpi_known and jpg.pages[0].dpi == 288


def test_a_broken_file_is_reported_not_raised(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.7\nthis is not a pdf")
    doc = load_document(bad)
    assert doc.error
    missing = load_document(tmp_path / "nope.pdf")
    assert missing.error
