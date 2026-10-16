"""Layer 4: what the ink's edges look like (a small learned model).

Patches are cut where ink meets paper, with the page first rescaled so the
pen is 4 px wide, and each patch is described by a few numbers a person can
name: how its darkness values are spread (a renderer's anti-aliasing comes
in steps; a scanner's is continuous), how sharp its edge is, how much fine
detail its outline carries (Scribe's letters are 144-dpi crops enlarged and
re-thresholded, so their outlines are smooth for their size), and how solid
the ink is inside the stroke. A logistic model over those numbers scores
each patch; a page's score is the mean.

This layer learns from whatever pages it is trained on, so it can learn the
difference between two renderers rather than between generated and written
ink. Its cross-source and degraded accuracy are in docs/results.md, and the
verdict lets it make a page "suspicious" at most, never "generated".
"""

from __future__ import annotations

from importlib import resources

import cv2
import numpy as np

from .ink import InkMap, extract_ink
from .layers import LayerResult
from .load import Document

MODEL_FILE = "texture.npz"
FEATURE_VERSION = 1
PATCH = 32
PEN_PX = 4.0
N_FEATURES = 44


def model_path():
    path = resources.files("inkcheck").joinpath("models", MODEL_FILE)
    return path if path.is_file() else None


def _normalised(ink: InkMap) -> tuple[np.ndarray, np.ndarray]:
    """Darkness and mask rescaled so the pen is PEN_PX wide."""
    f = PEN_PX / max(ink.stroke_width, 0.5)
    if abs(f - 1) < 0.03:
        return ink.darkness, ink.mask
    method = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    dark = cv2.resize(ink.darkness, None, fx=f, fy=f, interpolation=method).clip(0, 1)
    mask = cv2.resize(ink.mask.astype(np.uint8), None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST).astype(bool)
    return dark, mask


def patches(ink: InkMap, n: int = 160, rng=None) -> np.ndarray:
    """Up to `n` PATCH×PATCH darkness patches centred on ink edges (n×PATCH×PATCH)."""
    rng = rng or np.random.default_rng(0)
    dark, mask = _normalised(ink)
    edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(edge)
    h = PATCH // 2
    inside = (ys >= h) & (xs >= h) & (ys < dark.shape[0] - h) & (xs < dark.shape[1] - h)
    ys, xs = ys[inside], xs[inside]
    if len(ys) == 0:
        return np.zeros((0, PATCH, PATCH), np.float32)
    pick = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
    return np.stack([dark[y - h : y + h, x - h : x + h] for y, x in zip(ys[pick], xs[pick])]).astype(np.float32)


_RADIUS = None


def _radial_bands(size: int, bands: int = 8) -> np.ndarray:
    global _RADIUS
    if _RADIUS is None or _RADIUS[0] != size:
        y, x = np.indices((size, size)) - size / 2
        r = np.sqrt(x * x + y * y) / (size / 2)
        _RADIUS = (size, np.minimum((r * bands).astype(int), bands))
    return _RADIUS[1]


def features(batch: np.ndarray) -> np.ndarray:
    """Named numbers per patch (see the module notes); one row each."""
    if len(batch) == 0:
        return np.zeros((0, N_FEATURES), np.float32)
    rows = []
    bands = _radial_bands(PATCH)
    for p in batch:
        ink = p[p > 0.03]
        core = float(np.percentile(ink, 90)) if ink.size else 1.0
        q = np.clip(p / max(core, 1e-3), 0, 1)
        values = q[(q > 0.03) & (q < 0.97)]
        hist = np.histogram(values, bins=16, range=(0.03, 0.97))[0].astype(np.float32)
        hist /= max(hist.sum(), 1.0)
        # Steps: how much of the edge sits on a few repeated levels.
        fine = np.histogram(values, bins=64, range=(0.03, 0.97))[0].astype(np.float32)
        top = np.sort(fine)[::-1]
        steps = float(top[:4].sum() / max(fine.sum(), 1.0))
        gy, gx = np.gradient(q)
        g = np.sqrt(gx * gx + gy * gy)
        ghist = np.histogram(g[g > 0.02], bins=8, range=(0.02, 1.0))[0].astype(np.float32)
        ghist /= max(ghist.sum(), 1.0)
        spectrum = np.abs(np.fft.fftshift(np.fft.fft2(q - q.mean()))) ** 2
        power = np.array([spectrum[bands == b].mean() for b in range(8)], np.float32)
        power = np.log1p(power) - np.log1p(power.max())
        lap = cv2.Laplacian(q, cv2.CV_32F)
        solid = float(np.mean(q[q > 0.5] > 0.95)) if (q > 0.5).any() else 0.0
        rows.append(np.concatenate([hist, [steps, solid, float(np.std(ink / max(core, 1e-3))) if ink.size else 0.0,
                                           float(g.max()), float(np.mean(np.abs(lap))), float(np.mean(values)) if values.size else 0.0,
                                           float(values.size / max(ink.size, 1)), float(core)], ghist, power, [float(np.std(lap)),
                                                                                                      float(np.percentile(g, 95))],
                                    [float(np.mean(q > 0.5)), float(np.mean((q > 0.1) & (q < 0.9)))]]))
    return np.array(rows, np.float32)


def page_features(ink: InkMap, n: int = 160, seed: int = 0) -> np.ndarray:
    return features(patches(ink, n, np.random.default_rng(seed)))


class TextureModel:
    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.mean, self.scale = data["mean"], data["scale"]
        self.coef, self.intercept = data["coef"], float(data["intercept"])
        self.version = int(data["version"])
        self.notes = str(data["notes"]) if "notes" in data else ""

    def patch_probability(self, rows: np.ndarray) -> np.ndarray:
        z = (rows - self.mean) / self.scale
        return 1 / (1 + np.exp(-(z @ self.coef + self.intercept)))


def save_model(path, mean, scale, coef, intercept, notes: str) -> None:
    np.savez(path, mean=mean, scale=scale, coef=coef, intercept=np.float64(intercept), version=np.int64(FEATURE_VERSION),
             notes=np.str_(notes))


def assess_texture(doc: Document, inks: list[InkMap] | None = None) -> LayerResult:
    result = LayerResult("texture", None)
    path = model_path()
    if path is None:
        result.add("no-model", "No ink-texture model is installed, so this layer was skipped.")
        return result
    model = TextureModel(path)
    if model.version != FEATURE_VERSION:
        result.add("no-model", "The installed ink-texture model is out of date, so this layer was skipped.")
        return result
    inks = inks or [extract_ink(p.pixels, p.dpi) for p in doc.pages]
    probs = []
    for i, ink in enumerate(inks[:6]):
        if ink.mask.sum() < 500:
            continue
        rows = page_features(ink, seed=i)
        if len(rows):
            probs.append(model.patch_probability(rows))
    if not probs:
        result.add("too-little", "Too little ink to judge its texture.")
        return result
    p = np.concatenate(probs)
    result.score = float(p.mean())
    result.data.update(patches=int(len(p)), share_over_half=float(np.mean(p > 0.5)), notes=model.notes)
    weight = "moderate" if result.score >= 0.8 else "info"
    result.add("texture", f"The ink's edges look {'like Scribe-rendered ink' if result.score >= 0.5 else 'unlike Scribe-rendered ink'} "
               f"(model score {result.score:.2f} over {len(p)} edge patches). This check learns from the pages it was "
               "trained on and is easily fooled by printing and scanning, so it can only make a page worth a closer look.", weight)
    return result
