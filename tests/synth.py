"""Procedurally drawn pages for the tests.

Nothing here is real handwriting. A *glyph* is one or two smooth strokes in
a unit box. A person-style page draws every instance of a glyph afresh, with
its own small wobble of the strokes; a generator-style page draws each glyph
once to a bitmap and places affine copies of that bitmap, as Scribe does with
the crops from a practice sheet.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image

PAPER = (252, 252, 250)
RULE = (202, 216, 234)
INK = (43, 47, 82)
QUARTZ = "iOS Version 18.7.8 (Build 22H352) Quartz PDFContext"
PDF_LIB = "pdf-lib (https://github.com/Hopding/pdf-lib)"


def _catmull_rom(points: np.ndarray, per_segment: int = 16) -> np.ndarray:
    """A smooth curve through `points` (k×2)."""
    p = np.vstack([points[0], points, points[-1]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, per_segment, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(points[-1])
    return np.array(out)


def glyph_bank(seed: int, n: int = 30) -> list[list[np.ndarray]]:
    """`n` glyphs, each a list of strokes (control points in a 0.8×1 box)."""
    rng = np.random.default_rng(seed)
    bank = []
    for _ in range(n):
        strokes = []
        for _ in range(int(rng.integers(1, 3))):
            k = int(rng.integers(5, 8))
            strokes.append(np.column_stack([rng.uniform(0.05, 0.75, k), rng.uniform(0.05, 0.95, k)]))
        bank.append(strokes)
    return bank


def draw_glyph(strokes, size: float, width: float, wobble: float = 0.0, rng=None, supersample: int = 4) -> np.ndarray:
    """One instance as darkness (0..1), `size` px tall, pen `width` px."""
    pad = int(np.ceil(width * 2))
    h, w = int(size) + 2 * pad, int(size * 0.8) + 2 * pad
    canvas = np.zeros((h * supersample, w * supersample), np.uint8)
    for points in strokes:
        p = points + (rng.normal(0, wobble, points.shape) if wobble else 0)
        curve = (_catmull_rom(p) * size + pad) * supersample
        cv2.polylines(canvas, [np.round(curve).astype(np.int32)], False, 255,
                      thickness=max(1, int(round(width * supersample))), lineType=cv2.LINE_AA)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def _affine(rng, scale_spread=0.08, rotate_deg=2.5, shear=0.04) -> np.ndarray:
    s = 1 + rng.uniform(-scale_spread, scale_spread)
    a = np.deg2rad(rng.uniform(-rotate_deg, rotate_deg))
    k = rng.uniform(-shear, shear)
    return s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, k], [0, 1]])


def _stamp(page_dark: np.ndarray, patch: np.ndarray, x: float, y: float, linear: np.ndarray | None = None) -> None:
    """Puts `patch` with its top-left near (x, y), through `linear` (2×2) about its centre."""
    h, w = patch.shape
    if linear is None:
        linear = np.eye(2)
    cx, cy = w / 2, h / 2
    big = int(max(h, w) * 1.5) + 4
    m = np.zeros((2, 3))
    m[:, :2] = linear
    m[:, 2] = np.array([big / 2, big / 2]) - linear @ np.array([cx, cy])
    warped = cv2.warpAffine(patch, m, (big, big), flags=cv2.INTER_LINEAR)
    # Sub-pixel placement, as a renderer that positions glyphs freely would.
    fx, fy = x - np.floor(x), y - np.floor(y)
    warped = cv2.warpAffine(warped, np.float32([[1, 0, fx], [0, 1, fy]]), (big, big), flags=cv2.INTER_LINEAR)
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    H, W = page_dark.shape
    x1, y1 = min(W, x0 + big), min(H, y0 + big)
    if x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
        return
    region = page_dark[y0:y1, x0:x1]
    np.maximum(region, warped[: y1 - y0, : x1 - x0], out=region)


def _layout(rng, count: int, size: float, width_px: int, height_px: int, margin: int = 60):
    """Positions for `count` glyphs in lines, left to right."""
    advance, line = size * 0.8 + size * 0.45, size * 1.9
    x, y = margin, margin
    spots = []
    for _ in range(count):
        if x + advance > width_px - margin:
            x, y = margin, y + line
        if y + line > height_px - margin:
            break
        spots.append((x + rng.uniform(-2, 2), y + rng.uniform(-3, 3)))
        x += advance + rng.uniform(0, size * 0.3)
    return spots


def _order(rng, n_glyphs: int, per_glyph: tuple[int, int]) -> list[int]:
    order = [g for g in range(n_glyphs) for _ in range(int(rng.integers(per_glyph[0], per_glyph[1] + 1)))]
    rng.shuffle(order)
    return order


def to_rgb(dark: np.ndarray, ruled: bool = True, rule_every: int = 76, ink=INK, paper=PAPER) -> np.ndarray:
    page = np.empty(dark.shape + (3,), np.float32)
    page[:] = paper
    if ruled:
        for y in range(rule_every, dark.shape[0], rule_every):
            page[y : y + 2] = RULE
    d = dark[..., None]
    return np.clip(page * (1 - d) + np.array(ink, np.float32) * d, 0, 255).round().astype(np.uint8)


def human_page(seed: int = 1, n_glyphs: int = 30, per_glyph=(3, 6), size: float = 40, width: float = 3.2,
               wobble: float = 0.035, shape=(1200, 1600), ruled: bool = True, bank=None) -> np.ndarray:
    """Every instance drawn afresh: its own wobble and its own pose."""
    rng = np.random.default_rng(seed)
    bank = bank or glyph_bank(seed + 1000, n_glyphs)
    dark = np.zeros(shape, np.float32)
    order = _order(rng, len(bank), per_glyph)
    for g, (x, y) in zip(order, _layout(rng, len(order), size, shape[1], shape[0])):
        patch = draw_glyph(bank[g], size * (1 + rng.uniform(-0.06, 0.06)), width, wobble, rng)
        _stamp(dark, patch, x, y, _affine(rng))
    return to_rgb(dark, ruled)


def generated_page(seed: int = 2, n_glyphs: int = 30, per_glyph=(3, 6), size: float = 40, width: float = 3.2,
                   shape=(1200, 1600), ruled: bool = True, bank=None, wobble_once: float = 0.035) -> np.ndarray:
    """Each glyph drawn once to a bitmap, then placed as affine copies."""
    rng = np.random.default_rng(seed)
    bank = bank or glyph_bank(seed + 1000, n_glyphs)
    bitmaps = [draw_glyph(strokes, size, width, wobble_once, rng) for strokes in bank]
    dark = np.zeros(shape, np.float32)
    order = _order(rng, len(bank), per_glyph)
    for g, (x, y) in zip(order, _layout(rng, len(order), size, shape[1], shape[0])):
        _stamp(dark, bitmaps[g], x, y, _affine(rng))
    return to_rgb(dark, ruled)


def simple_marks_page(seed: int = 3, count: int = 160, size: float = 40, width: float = 3.2, shape=(1200, 1600)) -> np.ndarray:
    """Short bars, dots, dashes and brackets, drawn by hand: shapes anyone repeats closely."""
    rng = np.random.default_rng(seed)
    marks = [
        [np.array([[0.4, 0.1], [0.4, 0.9]])],                      # a stroke like 1 or l
        [np.array([[0.1, 0.5], [0.7, 0.5]])],                      # a dash
        [np.array([[0.4, 0.45], [0.42, 0.5]])],                    # a dot
        [np.array([[0.6, 0.05], [0.3, 0.2], [0.25, 0.5], [0.3, 0.8], [0.6, 0.95]])],  # a bracket
        [np.array([[0.1, 0.35], [0.7, 0.35]]), np.array([[0.1, 0.65], [0.7, 0.65]])],  # an equals sign
    ]
    dark = np.zeros(shape, np.float32)
    for (x, y) in _layout(rng, count, size, shape[1], shape[0]):
        patch = draw_glyph(marks[int(rng.integers(0, len(marks)))], size, width, 0.02, rng)
        _stamp(dark, patch, x, y, _affine(rng))
    return to_rgb(dark, True)


def add_printed_text(rgb: np.ndarray, lines: list[str], top: int = 40, left: int = 60, scale: float = 1.0) -> np.ndarray:
    """Typeset text (Hershey font: every copy of a letter identical), as on a printed worksheet."""
    out = rgb.copy()
    for i, line in enumerate(lines):
        cv2.putText(out, line, (left, top + int(i * 42 * scale)), cv2.FONT_HERSHEY_COMPLEX, scale, (20, 20, 20), 2, cv2.LINE_AA)
    return out


def lasso_copy(rgb: np.ndarray, box: tuple[int, int, int, int], to: tuple[int, int]) -> np.ndarray:
    """A block of the page copied elsewhere unchanged, as a note app's lasso copy does."""
    x, y, w, h = box
    out = rgb.copy()
    src = rgb[y : y + h, x : x + w]
    dst = out[to[1] : to[1] + h, to[0] : to[0] + w]
    np.minimum(dst, src, out=dst)
    return out


def photographed(rgb: np.ndarray, seed: int = 4, quality: int = 85) -> bytes:
    """The page as a phone photo: uneven light, sensor noise, JPEG."""
    rng = np.random.default_rng(seed)
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    light = 0.82 + 0.18 * (xx / w) * (1 - 0.4 * yy / h)
    noisy = rgb.astype(np.float32) * light[..., None] + rng.normal(0, 3.0, rgb.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8)).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def png_bytes(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


def image_pdf(path: Path, pages: list[bytes], producer: str = PDF_LIB, creator: str | None = PDF_LIB,
              size_pt=(612, 792)) -> Path:
    """A PDF whose pages are each one full-page image (PNG or JPEG bytes)."""
    doc = pymupdf.open()
    for data in pages:
        page = doc.new_page(width=size_pt[0], height=size_pt[1])
        page.insert_image(page.rect, stream=data)
    doc.set_metadata({"producer": producer, "creator": creator or "", "title": "", "author": ""})
    doc.save(path, deflate=True)
    return path


def vector_pdf(path: Path, seed: int = 5, size_pt=(612, 792), recognised_text: str = "x + 2 = 5 so x = 3") -> Path:
    """A note app's export: ink as vector strokes, plus an invisible recognised-text layer."""
    rng = np.random.default_rng(seed)
    bank = glyph_bank(seed + 1000, 20)
    doc = pymupdf.open()
    page = doc.new_page(width=size_pt[0], height=size_pt[1])
    shape = page.new_shape()
    x, y = 50.0, 60.0
    for _ in range(90):
        strokes = bank[int(rng.integers(0, len(bank)))]
        for points in strokes:
            curve = _catmull_rom(points + rng.normal(0, 0.035, points.shape)) * 14 + (x, y)
            shape.draw_polyline([pymupdf.Point(px, py) for px, py in curve])
            shape.finish(color=(0.08, 0.0, 0.6), width=1.1, closePath=False, lineCap=1, lineJoin=1)
        x += 16
        if x > size_pt[0] - 60:
            x, y = 50.0, y + 34
    shape.commit()
    page.insert_text((50, 40), recognised_text, fontsize=11, render_mode=3)
    doc.set_metadata({"producer": QUARTZ, "creator": "", "title": "", "author": ""})
    doc.save(path)
    return path
