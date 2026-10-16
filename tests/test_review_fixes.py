"""Edge cases an independent review found (one test per fix)."""

import io

import cv2
import numpy as np
import pymupdf
from PIL import Image

import synth
from inkcheck.cli import main
from inkcheck.load import load_document
from inkcheck.reuse import Match, extra_copies
from inkcheck.reuse import assess_reuse


def _match(a, b):
    return Match(a, b, 1.0, 0.01, 0.999, np.eye(2, 3), {"scale": 1.0, "rotation_deg": 0.0, "shear": 0.0, "aspect": 1.0})


def test_a_chance_pair_counts_as_one_extra_copy_and_a_cluster_as_its_size_minus_one():
    assert extra_copies([_match(0, 1)]) == 1
    assert extra_copies([_match(0, 1), _match(1, 2), _match(2, 3)]) == 3
    assert extra_copies([_match(0, 1), _match(5, 6)]) == 2


def test_a_resized_lasso_copy_is_one_block(tmp_path):
    page = synth.human_page(seed=71)
    block = page[50:130, 60:760]
    smaller = cv2.resize(block, None, fx=0.85, fy=0.85, interpolation=cv2.INTER_AREA)
    out = page.copy()
    h, w = smaller.shape[:2]
    np.minimum(out[1000 : 1000 + h, 600 : 600 + w], smaller, out=out[1000 : 1000 + h, 600 : 600 + w])
    (tmp_path / "p.png").write_bytes(synth.png_bytes(out))
    r = assess_reuse(load_document(tmp_path / "p.png"), workers=1)
    assert r.data["copied_blocks"] == 1
    assert r.data["extra_copies"] <= 1


def test_a_phone_jpeg_with_a_preview_is_one_page(tmp_path):
    rgb = synth.human_page(seed=72, shape=(800, 600))
    main_image = Image.fromarray(rgb)
    preview = main_image.resize((300, 400))
    path = tmp_path / "photo.jpg"
    main_image.save(path, "MPO", save_all=True, append_images=[preview])
    doc = load_document(path)
    assert len(doc.pages) == 1 and doc.pages[0].pixels.shape[:2] == (800, 600)


def test_a_16_bit_greyscale_page_keeps_its_ink(tmp_path):
    rgb = synth.generated_page(seed=73, shape=(600, 800))
    grey16 = (cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.uint16) * 257)
    path = tmp_path / "deep.png"
    Image.fromarray(grey16, mode="I;16").save(path)
    pixels = load_document(path).pages[0].pixels
    assert pixels.min() < 100 and pixels.max() > 200


def test_printed_text_boxes_follow_a_rotated_pdf_page(tmp_path):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((72, 100), "Solve each equation and show your work", fontsize=14)
    page.set_rotation(90)
    path = tmp_path / "rotated.pdf"
    doc.save(path)
    loaded = load_document(path).pages[0]
    h, w = loaded.pixels.shape[:2]
    assert w > h  # rendered sideways
    dark = loaded.pixels.min(axis=2) < 128
    x0, y0, x1, y1 = (int(v) for v in loaded.facts.text_boxes[0])
    inside = dark[max(0, y0) : y1, max(0, x0) : x1].sum()
    assert inside > 0.8 * dark.sum()


def test_reports_for_files_with_one_name_do_not_overwrite_each_other(tmp_path):
    for folder, seed in (("a", 74), ("b", 75)):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "hw.png").write_bytes(synth.png_bytes(synth.human_page(seed=seed, shape=(500, 700))))
    main(["scan", str(tmp_path / "a" / "hw.png"), str(tmp_path / "b" / "hw.png"), "--html", str(tmp_path / "out"), "-q",
          "--workers", "1"])
    assert len(list((tmp_path / "out").glob("*.html"))) == 2


def test_the_tiff_fallback_keeps_every_page(tmp_path, monkeypatch):
    import inkcheck.load as load

    pages = [synth.human_page(seed=s, shape=(300, 400)) for s in (76, 77, 78)]
    path = tmp_path / "multi.tif"
    cv2.imwritemulti(str(path), [cv2.cvtColor(p, cv2.COLOR_RGB2BGR) for p in pages])

    def broken(_):
        raise ValueError("decoder failed")

    monkeypatch.setattr(load, "_load_image_pil", broken)
    assert len(load_document(path).pages) == 3
