"""How often one person's repeated symbols look like copies: MathWriting, one expression at a time.

    python -I -m inkcheck.bench.mw_null --source <unpacked mathwriting-2024> [--expressions 4000] [--workers 12]

Each human expression is written by one person, so any twin found inside it
is that writer repeating a symbol (x … x, 2 … 2) closely enough to pass for a
copy. Pooled over thousands of writers, this is the same-writer twin rate
behind the binomial test, measured on tablet ink and on maths, which the
CVL scans are not. Writes runs/mw-null/summary.json.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import beta

from ..load import Document, Page, PageFacts
from ..reuse import at_threshold, extra_copies, find_twins
from .mathwriting import parse, render

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (0.06, 0.07, 0.08, 0.09, 0.10)


def score(args) -> dict:
    path, seed = args
    rng = np.random.default_rng(seed)
    strokes, label, _ = parse(Path(path))
    patch = render(strokes, float(rng.uniform(30, 52)), float(rng.uniform(2.6, 4.2)))
    if patch.size <= 1:
        return {"complex": 0}
    pad = 60
    dark = np.zeros((patch.shape[0] + 2 * pad, patch.shape[1] + 2 * pad), np.float32)
    dark[pad : pad + patch.shape[0], pad : pad + patch.shape[1]] = patch
    rgb = np.clip(252 * (1 - dark[..., None]) + np.array((30, 30, 60), np.float32) * dark[..., None], 0, 255).astype(np.uint8)
    doc = Document(Path(path).with_suffix(".png"), "image",
                   [Page(0, rgb, 288.0, PageFacts(images=1, image_codecs=["png"], image_px=(rgb.shape[1], rgb.shape[0]), full_page_image=True))])
    findings, _ = find_twins(doc, workers=1, threshold=max(THRESHOLDS))
    out = {"complex": findings.complex_count}
    for t in THRESHOLDS:
        twins = at_threshold(findings, t).twins
        out[f"{t:.2f}"] = extra_copies(twins)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True)
    parser.add_argument("--expressions", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    files = sorted(Path(args.source).joinpath("test").glob("*.inkml")) + sorted(Path(args.source).joinpath("valid").glob("*.inkml"))
    rng = np.random.default_rng(3)
    picked = [str(files[i]) for i in rng.choice(len(files), size=min(args.expressions, len(files)), replace=False)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(score, [(p, i) for i, p in enumerate(picked)], chunksize=8))
    n = sum(r["complex"] for r in rows)
    summary = {"expressions": len(rows), "complex": n}
    for t in THRESHOLDS:
        k = sum(r.get(f"{t:.2f}", 0) for r in rows)
        summary[f"{t:.2f}"] = {"extra_copies": k, "rate": k / max(n, 1), "upper95": float(beta.ppf(0.95, k + 1, n - k)) if n > k else 1.0}
    out = ROOT / "runs" / "mw-null"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
