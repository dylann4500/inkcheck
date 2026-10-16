"""Pages from Google's MathWriting dataset (CC BY-NC-SA 4.0, non-commercial).

MathWriting holds online handwriting: math expressions written by people
on touchscreens and digital pens (`train/`, `valid/`, `test/`), and
synthetic expressions stitched together from those people's individual
glyphs (`synthetic/`, from `symbols/`), a sample-assembling generator built
by someone other than Scribe.

    python -I -m inkcheck.bench.mathwriting --source <unpacked mathwriting-2024> --pages 120 [--out data/public/mathwriting-pages]

writes worksheet-like pages (about fifteen expressions each, ruled or blank,
288 dpi, a tablet pen's ink) of each kind, plus `pages.jsonl` for the
manifest. Human pages mix writers (the dataset names no writer), so their
twins can only come from one writer repeating a symbol inside one
expression; `--expressions N` also scores N single human expressions
alone, for that same-writer rate.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NS = "{http://www.w3.org/2003/InkML}"
INKS = ((44, 47, 82), (20, 20, 20), (21, 0, 154), (35, 60, 140))


def parse(path: Path) -> tuple[list[np.ndarray], str, str]:
    root = ET.parse(path).getroot()
    notes = {a.get("type"): (a.text or "") for a in root.iter(f"{NS}annotation")}
    strokes = []
    for trace in root.iter(f"{NS}trace"):
        points = []
        for chunk in (trace.text or "").split(","):
            parts = chunk.split()
            if len(parts) >= 2:
                points.append((float(parts[0]), float(parts[1])))
        if points:
            strokes.append(np.array(points, np.float64))
    return strokes, notes.get("normalizedLabel") or notes.get("label", ""), notes.get("inkCreationMethod", "")


def render(strokes: list[np.ndarray], symbol_px: float, pen_px: float, supersample: int = 4) -> np.ndarray:
    """One expression as darkness (0..1): scaled so a typical stroke spans `symbol_px`."""
    if not strokes:
        return np.zeros((1, 1), np.float32)
    extents = [max(np.ptp(s[:, 0]), np.ptp(s[:, 1])) for s in strokes]
    typical = float(np.median([e for e in extents if e > 0] or [1.0]))
    scale = symbol_px / max(typical, 1e-6)
    pts = np.vstack(strokes)
    origin = pts.min(axis=0)
    pad = int(np.ceil(pen_px * 2)) + 2
    width = int(np.ceil(np.ptp(pts[:, 0]) * scale)) + 2 * pad
    height = int(np.ceil(np.ptp(pts[:, 1]) * scale)) + 2 * pad
    if width * height > 6_000_000:  # a runaway expression; skip it
        return np.zeros((1, 1), np.float32)
    canvas = np.zeros((height * supersample, width * supersample), np.uint8)
    thickness = max(1, int(round(pen_px * supersample)))
    for s in strokes:
        xy = ((s - origin) * scale + pad) * supersample
        xy = np.round(xy).astype(np.int32)
        if len(xy) == 1:
            cv2.circle(canvas, tuple(xy[0]), thickness // 2, 255, -1, cv2.LINE_AA)
        else:
            cv2.polylines(canvas, [xy], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    return cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


def compose(expressions: list[np.ndarray], rng, ruled: bool, ink, size=(3168, 2448)) -> np.ndarray:
    """Expressions set down a page in rows, as a student works down a worksheet."""
    page_h, page_w = size
    dark = np.zeros(size, np.float32)
    y, left = 160, 180
    for patch in expressions:
        h, w = patch.shape
        if w > page_w - left - 120:
            f = (page_w - left - 120) / w
            patch = cv2.resize(patch, (int(w * f), max(1, int(h * f))), interpolation=cv2.INTER_AREA)
            h, w = patch.shape
        if y + h > page_h - 120:
            break
        x = left + int(rng.integers(0, 80))
        region = dark[y : y + h, x : x + w]
        np.maximum(region, patch, out=region)
        y += h + int(rng.integers(40, 110))
    page = np.empty(size + (3,), np.float32)
    page[:] = (252, 252, 250)
    if ruled:
        for ry in range(140, page_h, 96):
            page[ry : ry + 2] = (202, 216, 234)
    d = dark[..., None]
    return np.clip(page * (1 - d) + np.array(ink, np.float32) * d, 0, 255).round().astype(np.uint8)


def _inks(source: Path, folders: tuple[str, ...]) -> list[Path]:
    files = []
    for folder in folders:
        files += sorted((source / folder).glob("*.inkml"))
    return files


def build(source: Path, out: Path, pages: int, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    kinds = {"human": _inks(source, ("train", "valid", "test")), "synthetic": _inks(source, ("synthetic",))}
    for kind, files in kinds.items():
        if not files:
            continue
        folder = out / kind
        folder.mkdir(exist_ok=True)
        for n in range(pages):
            symbol_px = float(rng.uniform(30, 52))
            pen = float(rng.uniform(2.6, 4.2))
            picked = [files[int(i)] for i in rng.choice(len(files), size=min(18, len(files)), replace=False)]
            patches = [render(parse(p)[0], symbol_px, pen) for p in picked]
            patches = [p for p in patches if p.size > 1]
            page = compose(patches, rng, bool(rng.integers(0, 2)), INKS[int(rng.integers(0, len(INKS)))])
            path = folder / f"{kind}-{n:04d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(page, cv2.COLOR_RGB2BGR))
            rows.append({"id": f"mw-{kind}:{n:04d}", "path": str(path), "label": "real" if kind == "human" else "synthetic",
                         "source": f"mw-{kind}", "writer": "mixed" if kind == "human" else "mw-symbols",
                         "inks": [p.stem for p in picked]})
    (out / "pages.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return rows


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Worksheet-like pages from MathWriting.")
    parser.add_argument("--source", required=True, help="the unpacked mathwriting-2024 folder")
    parser.add_argument("--out", default=str(ROOT / "data" / "public" / "mathwriting-pages"))
    parser.add_argument("--pages", type=int, default=120)
    args = parser.parse_args(argv)
    rows = build(Path(args.source), Path(args.out), args.pages)
    print(f"{len(rows)} pages -> {args.out}")


if __name__ == "__main__":
    main()
