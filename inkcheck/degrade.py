"""What happens to a page on its way to a teacher, simulated.

`print-scan`: printed on a laser printer and scanned at 300 dpi (toner
darkens and spreads the ink, grain, a slight turn, scanner blur and noise,
JPEG). `photo`: a phone photo of the printed page (perspective, uneven
light, blur, sensor noise, colour cast, a random resolution, JPEG).
`jpeg`, `downscale` (half resolution) and `gray` are the single steps.
Every kind is deterministic for a given seed.
"""

from __future__ import annotations

import cv2
import numpy as np

KINDS = ("print-scan", "photo", "jpeg", "downscale", "gray")


def _jpeg(rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _low_frequency(rng, shape, cells: int = 6) -> np.ndarray:
    coarse = rng.normal(0, 1, (cells, cells)).astype(np.float32)
    field = cv2.resize(coarse, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    return field / max(float(np.abs(field).max()), 1e-6)


def _resize(img: np.ndarray, scale: float) -> np.ndarray:
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=method)


def print_scan(pixels: np.ndarray, dpi: float, rng) -> tuple[np.ndarray, float]:
    target = 300.0
    img = _resize(pixels, target / dpi)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    ink = 1 - grey
    ink = np.clip(ink * rng.uniform(1.15, 1.45), 0, 1)                 # toner prints darker than screen ink
    ink = cv2.GaussianBlur(ink, (0, 0), rng.uniform(0.35, 0.65))       # dot gain
    ink = np.clip(ink * (1 + rng.normal(0, 0.07, ink.shape).astype(np.float32)), 0, 1)  # toner grain
    printed = 1 - ink
    h, w = printed.shape
    turn = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-0.8, 0.8), 1 + rng.uniform(-0.004, 0.004))
    turn[:, 2] += rng.uniform(-6, 6, 2)
    scanned = cv2.warpAffine(printed, turn, (w, h), flags=cv2.INTER_LINEAR, borderValue=1.0)
    scanned = cv2.GaussianBlur(scanned, (0, 0), rng.uniform(0.45, 0.85))
    paper = 0.94 + 0.025 * _low_frequency(rng, scanned.shape)
    out = scanned * paper + rng.normal(0, 0.012, scanned.shape).astype(np.float32)
    rgb = np.repeat((np.clip(out, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=-1)
    return _jpeg(rgb, int(rng.integers(80, 93))), target


def photo(pixels: np.ndarray, dpi: float, rng) -> tuple[np.ndarray, float]:
    h0, w0 = pixels.shape[:2]
    width_px = rng.uniform(1400, 2600)                                   # the page's width in the photo
    scale = width_px / w0
    img = _resize(pixels, scale).astype(np.float32) / 255
    h, w = img.shape[:2]
    margin = int(0.05 * w)
    canvas_w, canvas_h = w + 2 * margin, h + 2 * margin
    table = np.array(rng.uniform(0.25, 0.45, 3), np.float32)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    jitter = rng.uniform(-0.035, 0.035, (4, 2)).astype(np.float32) * np.float32([w, h])
    dst = src + margin + jitter
    warp = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, warp, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                              borderValue=tuple(float(t) for t in table))
    yy, xx = np.mgrid[0:canvas_h, 0:canvas_w].astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    ramp = (np.cos(angle) * xx / canvas_w + np.sin(angle) * yy / canvas_h)
    ramp = (ramp - ramp.min()) / max(float(np.ptp(ramp)), 1e-6)
    vignette = 1 - 0.18 * (((xx / canvas_w - 0.5) ** 2 + (yy / canvas_h - 0.5) ** 2) * 2)
    light = (rng.uniform(0.72, 0.85) + rng.uniform(0.1, 0.22) * ramp) * vignette
    cast = np.array([1 + rng.uniform(-0.05, 0.05), 1.0, 1 + rng.uniform(-0.07, 0.03)], np.float32)
    out = out * light[..., None] * cast
    out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.6, 1.3))
    out = out + rng.normal(0, rng.uniform(0.008, 0.02), out.shape).astype(np.float32)
    rgb = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    return _jpeg(rgb, int(rng.integers(72, 89))), float(dpi * scale)


def degrade(pixels: np.ndarray, dpi: float, kind: str, seed: int = 0) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    if kind == "print-scan":
        return print_scan(pixels, dpi, rng)
    if kind == "photo":
        return photo(pixels, dpi, rng)
    if kind == "jpeg":
        return _jpeg(pixels, int(rng.integers(60, 86))), dpi
    if kind == "downscale":
        return _resize(pixels, 0.5), dpi / 2
    if kind == "gray":
        return np.repeat(cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)[..., None], 3, axis=-1), dpi
    raise ValueError(f"unknown degradation {kind!r}; choose from {', '.join(KINDS)}")


def degrade_document(doc, kind: str, seed: int = 0):
    """A copy of `doc` whose pages went through `kind` (each page its own seed),
    stored as images, as a scan or photo arrives: no PDF text layer, no metadata."""
    from pathlib import Path

    from .load import Document, Page, PageFacts

    if kind in (None, "", "none"):
        return doc
    lossy = kind in ("print-scan", "photo", "jpeg")
    pages = []
    for page in doc.pages:
        pixels, dpi = degrade(page.pixels, page.dpi, kind, seed + page.index)
        facts = PageFacts(images=1, image_codecs=["jpeg" if lossy else "png"], image_px=(pixels.shape[1], pixels.shape[0]),
                          full_page_image=True)
        pages.append(Page(page.index, pixels, dpi, facts, dpi_known=False))
    return Document(Path(f"{doc.path.stem}-{kind}{'.jpg' if lossy else '.png'}"), "image", pages, {})
