import numpy as np

import synth
from inkcheck.ink import components, extract_ink


LOOP = [np.array([[0.4 + 0.3 * np.cos(t), 0.5 + 0.42 * np.sin(t)] for t in np.linspace(0.3, 2 * np.pi, 9)])]


def _loops_on_ruled_paper(n=24, width=4.0, rule_every=60):
    """`n` separate loops, each crossed by a ruled line."""
    dark = np.zeros((600, 1000), np.float32)
    patch = synth.draw_glyph(LOOP, 80, width)
    for i in range(n):
        x, y = 30 + (i % 12) * 78, 80 + (i // 12) * 240
        h, w = patch.shape
        np.maximum(dark[y : y + h, x : x + w], patch, out=dark[y : y + h, x : x + w])
    return synth.to_rgb(dark, ruled=True, rule_every=rule_every)


def _true_width(width):
    line = synth.draw_glyph([np.array([[0.05, 0.5], [0.75, 0.5]])], 200, width)
    return line[:, 60:100].sum() / 40


def test_rules_are_removed_and_crossed_letters_stay_whole():
    rgb = _loops_on_ruled_paper()
    ink = extract_ink(rgb, 288)
    assert ink.rules_removed >= 5
    pieces = components(ink, 0)
    assert len(pieces) == 24
    # No ruled line survives in the mask: rows between loops are empty.
    assert not ink.mask[200:310].any()


def test_stroke_width_is_measured():
    for width in (2.0, 3.2, 5.0, 8.0):
        ink = extract_ink(_loops_on_ruled_paper(width=width), 288)
        assert abs(ink.stroke_width - _true_width(width)) / _true_width(width) < 0.15


def test_pieces_carry_their_own_ink_only():
    ink = extract_ink(synth.human_page(seed=7, shape=(600, 900)), 288)
    for piece in components(ink, 0)[:40]:
        assert piece.mask.any()
        assert piece.darkness.shape == piece.mask.shape
        assert piece.darkness[piece.mask].min() > 0
