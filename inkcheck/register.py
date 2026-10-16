"""Lining two pieces of ink up and measuring what is left.

The best affine map from one piece onto the other is found by ECC (OpenCV's
`findTransformECC`), coarse to fine on blurred darkness. What an affine map
cannot explain is the shape itself: the residual is measured between the two
pieces' centrelines (skeletons), in pen widths, so a heavier or lighter pen
does not count, and a shape a person wrote twice (never the same) stays
apart while two copies of one bitmap fall on top of each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import skeletonize

from .ink import Component


@dataclass
class Alignment:
    matrix: np.ndarray       # 2×3, maps page coordinates of a onto page coordinates of b
    correlation: float       # ECC's correlation after alignment (1 = identical)
    residual_mean: float     # mean centreline distance, pen widths
    residual_p90: float      # 90th percentile of it
    residual_max: float
    energy: float            # leftover darkness: 0 identical, 1 no overlap
    ok: bool
    # How far apart the two shapes' strokes lie after alignment, as an RMS
    # distance in pen widths, read from the leftover darkness against the
    # darkness gradient (resolution-independent; 0 for two copies of one bitmap).
    displacement: float = np.inf

    @staticmethod
    def failed() -> "Alignment":
        return Alignment(np.eye(2, 3), 0.0, np.inf, np.inf, np.inf, 1.0, False)


def _normalised(darkness: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    values = darkness[mask] if mask is not None and mask.any() else darkness[darkness > 0]
    core = float(np.percentile(values, 90)) if values.size else 1.0
    return np.clip(darkness / max(core, 1e-3), 0, 1).astype(np.float32)


def _centroid(img: np.ndarray) -> tuple[float, float]:
    total = float(img.sum())
    if total <= 0:
        return img.shape[1] / 2, img.shape[0] / 2
    ys, xs = np.indices(img.shape)
    return float((xs * img).sum() / total), float((ys * img).sum() / total)


def _place(img: np.ndarray, side: int) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """`img` on a side×side canvas with its centroid near the middle (integer shift)."""
    cx, cy = _centroid(img)
    ox, oy = int(round(side / 2 - cx)), int(round(side / 2 - cy))
    canvas = np.zeros((side, side), np.float32)
    h, w = img.shape
    ys0, xs0 = max(0, oy), max(0, ox)
    ye, xe = min(side, oy + h), min(side, ox + w)
    canvas[ys0:ye, xs0:xe] = img[ys0 - oy : ye - oy, xs0 - ox : xe - ox]
    return canvas, (ox, oy), (cx + ox, cy + oy)


def align_crops(a: np.ndarray, b: np.ndarray, stroke_width: float, a_origin=(0, 0), b_origin=(0, 0),
                a_mask: np.ndarray | None = None, b_mask: np.ndarray | None = None,
                early_exit: float | None = None) -> Alignment:
    """Aligns darkness crop `a` onto crop `b` (each with its page origin).

    With `early_exit`, a pair whose coarse correlation stays below it is not
    refined (it cannot be a twin) and comes back with an infinite displacement."""
    a = _normalised(a, a_mask)
    b = _normalised(b, b_mask)
    # Thin strokes (old 2x exports, small photos) are enlarged first, so the
    # alignment and the measures work on strokes at least ~3 px wide; very big
    # pieces (joined words at 4x) are shrunk to keep the work bounded, never
    # below 2 px strokes.
    sw0 = max(stroke_width, 1e-3)
    up = float(np.clip(3.0 / sw0, 1.0, 4.0))
    biggest = max(a.shape + b.shape)
    if biggest * up > 240:
        up = max(240.0 / biggest, min(up, 2.0 / sw0))
    if abs(up - 1) > 0.05:
        method = cv2.INTER_CUBIC if up > 1 else cv2.INTER_AREA
        a = cv2.resize(a, None, fx=up, fy=up, interpolation=method).clip(0, 1)
        b = cv2.resize(b, None, fx=up, fy=up, interpolation=method).clip(0, 1)
    else:
        up = 1.0
    sw = stroke_width * up
    side = int(np.ceil(max(a.shape + b.shape) * 1.5)) + 8
    A, a_shift, a_c = _place(a, side)
    B, b_shift, b_c = _place(b, side)
    # Initial warp maps B's canvas onto A's: centroids together, scale by size.
    scale = float(np.sqrt(max(a.sum(), 1e-6) / max(b.sum(), 1e-6)))
    warp = np.array([[scale, 0, a_c[0] - scale * b_c[0]], [0, scale, a_c[1] - scale * b_c[1]]], np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-6)
    correlation = 0.0
    try:
        for stage, sigma in enumerate((max(1.0, 0.9 * sw), max(0.5, 0.35 * sw))):
            blur_a = cv2.GaussianBlur(A, (0, 0), sigma)
            blur_b = cv2.GaussianBlur(B, (0, 0), sigma)
            correlation, warp = cv2.findTransformECC(blur_b, blur_a, warp, cv2.MOTION_AFFINE, criteria, None, 1)
            if stage == 0 and early_exit is not None and correlation < early_exit:
                return Alignment(_page_matrix(warp, a_origin, a_shift, b_origin, b_shift, up), float(correlation),
                                 np.inf, np.inf, np.inf, 1.0, True, np.inf)
    except cv2.error:
        return Alignment.failed()
    warped_a = cv2.warpAffine(A, warp, (side, side), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
    mean, p90, worst = _centreline_residual(warped_a, B, sw)
    energy = float(((warped_a - B) ** 2).sum() / max((warped_a**2).sum() + (B**2).sum(), 1e-9))
    return Alignment(_page_matrix(warp, a_origin, a_shift, b_origin, b_shift, up), float(correlation), mean, p90, worst,
                     energy, bool(np.isfinite(p90)), _displacement(warped_a, B, sw))


def _displacement(a: np.ndarray, b: np.ndarray, sw: float) -> float:
    """RMS distance between two aligned shapes, in pen widths.

    For a small displacement field u, b ≈ a + ∇a·u, so |b − a| over |∇a|
    estimates |u|. Both are blurred by half a pen width first, which keeps the
    estimate in its linear range and smooths away anti-aliasing noise."""
    sigma = max(0.5, 0.5 * sw)
    ia, ib = cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)
    gy_a, gx_a = np.gradient(ia)
    gy_b, gx_b = np.gradient(ib)
    gradient = 0.5 * float((gx_a**2 + gy_a**2 + gx_b**2 + gy_b**2).sum())
    if gradient <= 0:
        return np.inf
    return float(np.sqrt(((ia - ib) ** 2).sum() / gradient)) / max(sw, 1e-6)


def _centreline_residual(a: np.ndarray, b: np.ndarray, stroke_width: float) -> tuple[float, float, float]:
    sa, sb = skeletonize(a > 0.5), skeletonize(b > 0.5)
    if not sa.any() or not sb.any():
        return np.inf, np.inf, np.inf
    to_b = cv2.distanceTransform((~sb).astype(np.uint8), cv2.DIST_L2, 5)
    to_a = cv2.distanceTransform((~sa).astype(np.uint8), cv2.DIST_L2, 5)
    d = np.concatenate([to_b[sa], to_a[sb]]) / max(stroke_width, 1e-6)
    return float(d.mean()), float(np.percentile(d, 90)), float(d.max())


def _page_matrix(warp, a_origin, a_shift, b_origin, b_shift, up: float = 1.0) -> np.ndarray:
    """ECC's warp (B canvas → A canvas) as a map from a's page coordinates to b's."""
    def canvas_to_page(origin, shift):
        # canvas → enlarged crop (minus shift) → crop (÷ up) → page (+ origin)
        m = np.diag([1 / up, 1 / up, 1.0])
        m[0, 2] = origin[0] - shift[0] / up
        m[1, 2] = origin[1] - shift[1] / up
        return m

    w = np.vstack([warp.astype(np.float64), [0, 0, 1]])
    b_page_to_a_page = canvas_to_page(a_origin, a_shift) @ w @ np.linalg.inv(canvas_to_page(b_origin, b_shift))
    return np.linalg.inv(b_page_to_a_page)[:2]


def align(a: Component, b: Component, stroke_width: float, early_exit: float | None = None) -> Alignment:
    return align_crops(a.darkness, b.darkness, stroke_width, a.origin, b.origin, a.mask, b.mask, early_exit)


def transform_parts(matrix: np.ndarray) -> dict:
    """Scale, turn (degrees), shear and aspect of an affine map's linear part."""
    linear = np.asarray(matrix, np.float64)[:2, :2]
    q, r = np.linalg.qr(linear)
    # Keep r's diagonal positive so the rotation is the real one.
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    q, r = q * signs, (r.T * signs).T
    return {
        "scale": float(np.sqrt(abs(np.linalg.det(linear)))),
        "rotation_deg": float(np.degrees(np.arctan2(q[1, 0], q[0, 0]))),
        "shear": float(r[0, 1] / r[1, 1]) if r[1, 1] else 0.0,
        "aspect": float(r[1, 1] / r[0, 0]) if r[0, 0] else 1.0,
        "translation": (float(matrix[0][2]), float(matrix[1][2])),
    }
