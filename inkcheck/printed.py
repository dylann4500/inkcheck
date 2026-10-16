"""Is this piece of ink printed type or handwriting?

The reuse test sets printed type aside because type repeats exactly. So do
the letters a sample-assembled hand copies whole from its sentence library
(Scribe's "real copies, never jittered"), so how twins repeat cannot be the
only test. This model looks at the shape itself: each piece scaled into a
32×32 box (aspect kept), its HOG, and a few named numbers (stroke contrast
along the centreline, ink share, ends and junctions, size against the
page's letters).

Trained by inkcheck.bench.train_printed on letters set in the Mac's ordinary
text faces (handwriting-style fonts deliberately left out: text set in one
is generated handwriting) against handwriting from Scribe pages and real
pages, under the same degradations as the benchmark.
"""

from __future__ import annotations

from importlib import resources

import cv2
import numpy as np
from scipy.ndimage import convolve
from skimage.feature import hog
from skimage.morphology import skeletonize

from .ink import Component

MODEL_FILE = "printed.npz"
FEATURE_VERSION = 1
SIDE = 32
_NEIGHBOURS = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)


def vector(c: Component, stroke_width: float, letter_height: float) -> np.ndarray:
    x0, y0 = c.origin
    x, y, w, h = c.box
    crop = c.darkness[y - y0 : y - y0 + h, x - x0 : x - x0 + w]
    mask = c.mask[y - y0 : y - y0 + h, x - x0 : x - x0 + w]
    scale = (SIDE - 4) / max(h, w, 1)
    small = cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    box = np.zeros((SIDE, SIDE), np.float32)
    oy, ox = (SIDE - small.shape[0]) // 2, (SIDE - small.shape[1]) // 2
    box[oy : oy + small.shape[0], ox : ox + small.shape[1]] = small
    features = hog(box, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    skeleton = skeletonize(mask)
    if skeleton.any():
        widths = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)[skeleton] * 2
        contrast = float(np.std(widths) / max(np.mean(widths), 1e-6))
        mean_width = float(np.mean(widths) / max(stroke_width, 1e-6))
        counts = convolve(skeleton.astype(np.uint8), _NEIGHBOURS, mode="constant") * skeleton
        ends, junctions = float(np.count_nonzero(counts == 1)), float(np.count_nonzero(counts >= 3))
    else:
        contrast = mean_width = ends = junctions = 0.0
    extras = [h / max(w, 1), h / max(letter_height, 1), w / max(letter_height, 1), contrast, mean_width,
              float(mask.mean()), min(ends, 8) / 8, min(junctions, 8) / 8]
    return np.concatenate([features, extras]).astype(np.float32)


def model_path():
    path = resources.files("inkcheck").joinpath("models", MODEL_FILE)
    return path if path.is_file() else None


class PrintedModel:
    """A one-hidden-layer network (ReLU, logistic output), run in numpy."""

    def __init__(self, path):
        data = np.load(path, allow_pickle=False)
        self.mean, self.scale = data["mean"], data["scale"]
        self.w1, self.b1, self.w2, self.b2 = data["w1"], data["b1"], data["w2"], data["b2"]
        self.version = int(data["version"])

    def probability(self, rows: np.ndarray) -> np.ndarray:
        z = (rows - self.mean) / self.scale
        hidden = np.maximum(z @ self.w1 + self.b1, 0)
        return 1 / (1 + np.exp(-(hidden @ self.w2 + self.b2).ravel()))


_MODEL: PrintedModel | None = None


def load_model() -> PrintedModel | None:
    global _MODEL
    if _MODEL is None:
        path = model_path()
        if path is not None:
            model = PrintedModel(path)
            if model.version == FEATURE_VERSION:
                _MODEL = model
    return _MODEL


def typeset_probability(pieces, indices) -> dict[int, float]:
    """P(printed type) for the given piece indices; 0.5 for all when no model is installed.

    Each piece's answer is kept on the piece, since twins are sorted again at
    every threshold the benchmark tries and a shape's answer never changes."""
    model = load_model()
    indices = sorted(set(indices))
    if model is None or not indices:
        return {i: 0.5 for i in indices}
    todo = [i for i in indices if getattr(pieces[i], "typeset", None) is None]
    if todo:
        rows = np.stack([vector(pieces[i].comp, pieces[i].stroke_width, pieces[i].letter_height) for i in todo])
        for i, p in zip(todo, model.probability(rows).tolist()):
            pieces[i].typeset = p
    return {i: pieces[i].typeset for i in indices}


def save_model(path, mean, scale, network) -> None:
    """Saves a fitted sklearn MLPClassifier (one hidden layer) as plain arrays."""
    (w1, w2), (b1, b2) = network.coefs_, network.intercepts_
    np.savez(path, mean=mean, scale=scale, w1=w1, b1=b1, w2=w2, b2=b2, version=np.int64(FEATURE_VERSION))
