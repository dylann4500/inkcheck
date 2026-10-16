import cv2
import numpy as np
import pytest

import synth
from inkcheck.load import load_document
from inkcheck.reuse import assess_reuse


def _doc(tmp_path, rgb, name="page.png"):
    (tmp_path / name).write_bytes(synth.png_bytes(rgb))
    return load_document(tmp_path / name)


def _reuse(tmp_path, rgb):
    return assess_reuse(_doc(tmp_path, rgb), workers=1)


def test_a_generated_page_shows_reuse(tmp_path):
    r = _reuse(tmp_path, synth.generated_page(seed=21))
    assert r.data["complex"] >= 40
    assert r.data["duplicated"] / r.data["complex"] > 0.6
    assert r.score >= 0.9
    assert any(e["kind"] == "reused" and len(e["members"]) >= 3 for e in r.evidence)


def test_a_person_style_page_shows_none(tmp_path):
    r = _reuse(tmp_path, synth.human_page(seed=22))
    assert r.data["complex"] >= 40
    assert r.data["duplicated"] == 0
    assert r.score < 0.1


def test_printed_worksheet_text_is_not_reuse(tmp_path):
    page = synth.human_page(seed=23)
    page = synth.add_printed_text(page, ["Solve each equation and show", "your answers as nice as possible.",
                                         "Assessment: sessions and lessons"], top=60, scale=1.6)
    r = _reuse(tmp_path, page)
    assert r.data["duplicated"] <= 2
    assert r.score < 0.5


def test_a_lasso_copied_line_is_one_block_not_reuse(tmp_path):
    page = synth.human_page(seed=24)
    copied = synth.lasso_copy(page, (60, 50, 700, 80), (700, 1040))
    r = _reuse(tmp_path, copied)
    assert r.data["copied_blocks"] == 1
    assert r.data["duplicated"] <= 2
    assert r.score < 0.5


def test_too_low_a_resolution_is_said_not_guessed(tmp_path):
    page = synth.generated_page(seed=25)
    small = cv2.resize(page, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    r = _reuse(tmp_path, small)
    assert r.score is None
    assert r.data["too_little_reason"] in ("resolution", "few-shapes")


def test_simple_marks_never_count(tmp_path):
    r = _reuse(tmp_path, synth.simple_marks_page(seed=26))
    assert r.data["duplicated"] == 0


def test_a_scanned_printed_worksheet_is_not_reuse(tmp_path):
    from inkcheck.degrade import degrade

    page = synth.add_printed_text(synth.human_page(seed=27), ["Solve each equation and show", "your answers as nice as possible.",
                                                             "Assessment: sessions and lessons"], top=60, scale=1.6)
    scanned, _ = degrade(page, 288, "print-scan", seed=5)
    r = _reuse(tmp_path, scanned)
    assert r.data["duplicated"] <= 2
    assert (r.score or 0) < 0.5


def test_rigid_fragments_amid_print_are_print_but_scattered_ones_are_not():
    from inkcheck.reuse import Match, Piece, _fragments_amid_print
    from inkcheck.ink import Component

    def piece(x, y):
        c = Component(0, 0, (x - 5, y - 5, 10, 10), (0, 0), np.ones((10, 10), bool), np.ones((10, 10), np.float32), 100,
                      (float(x), float(y)))
        return Piece(c, None, 2.0, 20.0)

    rigid = {"scale": 1.0, "rotation_deg": 0.0, "shear": 0.0, "aspect": 1.0}
    # Printed letters in a paragraph at the top; two rigid twins inside it, two far below in the writing.
    pieces = [piece(100 + 25 * k, 100) for k in range(8)] + [piece(140, 130), piece(240, 130), piece(140, 900), piece(400, 900)]
    printed = set(range(8))
    twins = [Match(8, 9, 1.0, 0.03, 0.99, np.eye(2, 3), rigid), Match(10, 11, 1.0, 0.03, 0.99, np.eye(2, 3), rigid)]
    assert _fragments_amid_print(pieces, twins, printed) == {8, 9}
