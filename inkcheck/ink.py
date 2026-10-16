"""From page pixels to pieces of ink.

The page's own paper is estimated with a coarse median (so uneven light and
coloured paper drop out), darkness is measured against it, and ruled or grid
lines are taken out by subtracting what a long thin opening finds (a top-hat),
so ink that crosses a line keeps its darkness while the line itself goes.
What is left is thresholded with hysteresis and cut into connected pieces.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize


@dataclass
class InkMap:
    darkness: np.ndarray  # float32 0..1, rules removed
    mask: np.ndarray      # bool, the ink
    stroke_width: float   # px
    letter_height: float  # px, the typical height of a piece of writing
    rules_removed: int    # how many ruled or grid lines were taken out
    threshold: float


@dataclass
class Component:
    page: int
    id: int
    box: tuple[int, int, int, int]  # x, y, w, h in page pixels (unpadded)
    origin: tuple[int, int]         # page position of the crops' top-left
    mask: np.ndarray                # bool crop, this piece only
    darkness: np.ndarray            # float32 crop, this piece only
    area: int
    centroid: tuple[float, float]   # page pixels
    printed: bool = False           # overlaps typeset text the PDF itself holds


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114).astype(np.float32) / 255.0


def paper_luminance(lum: np.ndarray) -> np.ndarray:
    """The paper under every pixel, from a coarse median that ignores ink and rules."""
    h, w = lum.shape
    factor = max(1, int(round(max(h, w) / 400)))
    small = cv2.resize(lum, (max(1, w // factor), max(1, h // factor)), interpolation=cv2.INTER_AREA)
    # Ink is a minority of any window this size; the median sits on paper.
    k = 2 * (max(5, int(round(min(small.shape) / 25))) // 2) + 1
    paper = cv2.medianBlur((small * 255).astype(np.uint8), min(k, 255)).astype(np.float32) / 255.0
    paper = cv2.dilate(paper, np.ones((3, 3), np.uint8))  # never darker than nearby paper
    return np.maximum(cv2.resize(paper, (w, h), interpolation=cv2.INTER_LINEAR), 1e-3)


def remove_rules(darkness: np.ndarray) -> tuple[np.ndarray, int]:
    """Takes out long thin horizontal and vertical lines; returns the darkness and how many went."""
    h, w = darkness.shape
    out = darkness
    removed = 0
    for horizontal in (True, False):
        length = max(41, int((w if horizontal else h) * 0.12))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1) if horizontal else (1, length))
        lines = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        if float(lines.max(initial=0)) < 0.04:
            continue
        # Count the lines (bands of rows or columns where an opening survives).
        profile = (lines > 0.04).mean(axis=1 if horizontal else 0) > 0.2
        removed += int(np.count_nonzero(np.diff(profile.astype(np.int8)) == 1) + (1 if profile[0] else 0))
        out = np.clip(out - lines, 0, 1)
    return out, removed


def _otsu(values: np.ndarray) -> float:
    if values.size < 16:
        return 0.4
    hist, edges = np.histogram(values, bins=128, range=(0, 1))
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    total = weight[-1]
    mean = np.cumsum(hist * centres)
    between = (mean[-1] * weight - mean * total) ** 2 / np.maximum(weight * (total - weight), 1e-9)
    return float(centres[int(np.argmax(between))])


def stroke_width_of(mask: np.ndarray, darkness: np.ndarray | None = None) -> float:
    """Typical pen width: the ink's area over the length of its centreline.

    With `darkness`, the area counts anti-aliased edge pixels by their share
    of ink (against the strokes' own core darkness), so thin strokes on a
    low-resolution page are not under-measured by the threshold."""
    if np.count_nonzero(mask) < 16:
        return 2.0
    if darkness is None:
        area = float(np.count_nonzero(mask))
    else:
        near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        core = float(np.percentile(darkness[mask], 90)) or 1.0
        area = float(np.clip(darkness[near] / core, 0, 1).sum())
    length = _centreline_length(skeletonize(mask))
    return float(area / length) if length else 2.0


def _centreline_length(skeleton: np.ndarray) -> float:
    """Length of a one-pixel skeleton: straight steps count 1, diagonal steps sqrt 2, each counted once."""
    s = skeleton.astype(np.uint8)
    straight = np.count_nonzero(s[:, 1:] & s[:, :-1]) + np.count_nonzero(s[1:, :] & s[:-1, :])
    diagonal = np.count_nonzero(s[1:, 1:] & s[:-1, :-1]) + np.count_nonzero(s[1:, :-1] & s[:-1, 1:])
    # A diagonal step that also has a straight route through a corner pixel is counted by both; this
    # is close enough for a width that only sets scales.
    return float(straight + diagonal * np.sqrt(2)) or float(np.count_nonzero(s))


def extract_ink(pixels: np.ndarray, dpi: float) -> InkMap:
    lum = _luminance(pixels)
    darkness = np.clip(1.0 - lum / paper_luminance(lum), 0, 1).astype(np.float32)
    darkness, rules = remove_rules(darkness)
    candidates = darkness[darkness > 0.08]
    threshold = float(np.clip(_otsu(candidates), 0.22, 0.55))
    strong = darkness > threshold
    weak = darkness > threshold * 0.6
    count, labels = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(count, bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    mask = keep[labels]
    stroke = stroke_width_of(mask, darkness)
    letter = _letter_height(mask, stroke)
    return InkMap(darkness, mask, stroke, letter, rules, threshold)


def _letter_height(mask: np.ndarray, stroke: float) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas = stats[1:, cv2.CC_STAT_AREA]
    useful = heights[(heights >= 3 * stroke) & (areas >= 4 * stroke * stroke)]
    return float(np.median(useful)) if useful.size else 10 * stroke


def components(ink: InkMap, page: int, text_boxes=(), *, min_area_sw2: float = 1.5, max_height_letters: float = 6.0,
               max_width_letters: float = 12.0) -> list[Component]:
    """The connected pieces of ink, specks and big figures left out."""
    mask8 = ink.mask.astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask8, connectivity=8)
    sw, lh = ink.stroke_width, ink.letter_height
    pad = int(np.ceil(2 * sw)) + 2
    H, W = mask8.shape
    out = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        if area < min_area_sw2 * sw * sw or h > max_height_letters * lh or w > max_width_letters * lh:
            continue
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
        own = labels[y0:y1, x0:x1] == label
        near = cv2.dilate(own.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        dark = np.where(near, ink.darkness[y0:y1, x0:x1], 0).astype(np.float32)
        printed = any(_overlap((x, y, w, h), box) > 0.5 for box in text_boxes)
        out.append(Component(page, label, (x, y, w, h), (x0, y0), own, dark, area,
                             (float(centroids[label][0]), float(centroids[label][1])), printed))
    return out


def _overlap(box, text_box) -> float:
    x, y, w, h = box
    tx0, ty0, tx1, ty1 = text_box
    ix = max(0.0, min(x + w, tx1) - max(x, tx0))
    iy = max(0.0, min(y + h, ty1) - max(y, ty0))
    return ix * iy / max(w * h, 1)
