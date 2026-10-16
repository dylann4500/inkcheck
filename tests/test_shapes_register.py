import numpy as np

import synth
from inkcheck.ink import Component
from inkcheck.register import align_crops, transform_parts
from inkcheck.reuse import complexity
from inkcheck.shapes import describe


def _component(dark: np.ndarray) -> Component:
    mask = dark > 0.5
    ys, xs = np.nonzero(mask)
    box = (int(xs.min()), int(ys.min()), int(np.ptp(xs) + 1), int(np.ptp(ys) + 1))
    return Component(0, 1, box, (0, 0), mask, dark.astype(np.float32), int(mask.sum()), (float(xs.mean()), float(ys.mean())))


def _stamped(patch, linear=None, at=(10.3, 10.6)):
    canvas = np.zeros((110, 110), np.float32)
    synth._stamp(canvas, patch, at[0], at[1], linear)
    return canvas


def test_straight_strokes_and_dots_are_not_complex_but_curly_glyphs_are():
    bar = _component(synth.draw_glyph([np.array([[0.4, 0.05], [0.42, 0.95]])], 60, 4))
    dot = _component(synth.draw_glyph([np.array([[0.4, 0.5], [0.42, 0.52]])], 60, 4))
    curl = _component(synth.draw_glyph(synth.glyph_bank(3, 1)[0], 60, 4))
    for piece, expect in ((bar, False), (dot, False), (curl, True)):
        info = describe(piece, 4.8)
        assert (complexity(info) >= 16) == expect


def test_an_affine_copy_has_the_nearer_descriptor():
    bank = synth.glyph_bank(5, 2)
    base = synth.draw_glyph(bank[0], 60, 4)
    copy = _component(_stamped(base, synth._affine(np.random.default_rng(1))))
    first = _component(_stamped(base))
    other = _component(_stamped(synth.draw_glyph(bank[1], 60, 4)))
    d = lambda a, b: float(describe(a, 4.8).descriptor @ describe(b, 4.8).descriptor)
    assert d(first, copy) > d(first, other)


def test_alignment_undoes_a_nudge_and_measures_what_is_left():
    rng = np.random.default_rng(2)
    glyph = synth.glyph_bank(9, 1)[0]
    base = synth.draw_glyph(glyph, 50, 3.2)
    turn, scale, shear = np.deg2rad(2.0), 1.07, 0.03
    linear = scale * np.array([[np.cos(turn), -np.sin(turn)], [np.sin(turn), np.cos(turn)]]) @ np.array([[1, shear], [0, 1]])
    a, b = _stamped(base), _stamped(base, linear, at=(12.7, 9.2))
    copy = align_crops(a, b, 3.9)
    assert copy.displacement < 0.08
    parts = transform_parts(copy.matrix)
    assert abs(parts["scale"] - scale) < 0.02 and abs(parts["rotation_deg"] - 2.0) < 0.5
    redrawn = align_crops(_stamped(synth.draw_glyph(glyph, 50, 3.2, 0.035, rng)), _stamped(synth.draw_glyph(glyph, 50, 3.2, 0.035, rng)), 3.9)
    assert redrawn.displacement > 0.12
