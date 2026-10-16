"""What each piece of ink looks like, in numbers.

Two jobs: say how much a shape can reveal (a dot or a straight bar looks the
same every time anyone writes it, so it can never show reuse), and give each
shape a cheap descriptor so likely twins can be found without aligning every
pair on the page.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize

from .ink import Component, _centreline_length

DESCRIPTOR_SIDE = 24
# Shapes shorter than this many pen widths of centreline are left out of
# the reuse test. Set from real pages by eval/calibrate.py (calibration.json).
COMPLEX_MIN = 8.0


@dataclass
class ShapeInfo:
    skeleton_len: float  # px
    length_sw: float     # centreline length in pen widths
    endpoints: int
    junctions: int
    tortuosity: float    # centreline length over the box diagonal
    complexity: float
    aspect: float        # height / width of the box
    descriptor: np.ndarray


_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def describe(c: Component, stroke_width: float) -> ShapeInfo:
    skeleton = skeletonize(c.mask)
    length = _centreline_length(skeleton)
    counts = convolve(skeleton.astype(np.uint8), _NEIGHBOURS, mode="constant") * skeleton
    endpoints = int(np.count_nonzero(counts == 1))
    junctions = int(np.count_nonzero(counts >= 3))
    x, y, w, h = c.box
    diagonal = float(np.hypot(w, h)) or 1.0
    length_sw = length / max(stroke_width, 1e-6)
    return ShapeInfo(length, length_sw, endpoints, junctions, length / diagonal, length_sw,
                     h / max(w, 1), descriptor(c))


def descriptor(c: Component) -> np.ndarray:
    """The ink stretched to a fixed square and blurred: near-identical for affine
    twins with small turns and slants, unlike for different shapes."""
    x0, y0 = c.origin
    x, y, w, h = c.box
    crop = c.darkness[y - y0 : y - y0 + h, x - x0 : x - x0 + w]
    if crop.size == 0:
        return np.zeros(DESCRIPTOR_SIDE * DESCRIPTOR_SIDE, np.float32)
    side = DESCRIPTOR_SIDE - 4
    small = cv2.resize(crop, (side, side), interpolation=cv2.INTER_AREA)
    small = cv2.copyMakeBorder(small, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    v = small.ravel().astype(np.float32)
    v -= v.mean()
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def is_complex(info: ShapeInfo, minimum: float = COMPLEX_MIN) -> bool:
    return info.complexity >= minimum
