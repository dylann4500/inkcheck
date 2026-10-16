"""The evidence figure: copies on a generated page, against the closest pair on a real one.

    .venv/bin/python -I docs/paper/make_evidence.py

Uses MathWriting pages only (CC BY-NC-SA; Gervais et al., 2024): a page
stitched from its glyph pool, and a page of its contributors' handwriting.
Writes figures/evidence.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from inkcheck.load import load_document
from inkcheck.register import align
from inkcheck.report import _crop, _to_height, pair_overlay
from inkcheck.reuse import candidate_pairs, collect
from inkcheck.scan import scan_document

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def label(img: np.ndarray, text: str) -> np.ndarray:
    bar = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def hstack(images, gap=8) -> np.ndarray:
    h = max(i.shape[0] for i in images)
    padded = [np.vstack([i, np.full((h - i.shape[0], i.shape[1], 3), 255, np.uint8)]) for i in images]
    spacer = np.full((h, gap, 3), 255, np.uint8)
    out = padded[0]
    for i in padded[1:]:
        out = np.hstack([out, spacer, i])
    return out


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pages = [json.loads(l) for l in (ROOT / "data" / "public" / "mathwriting-pages" / "pages.jsonl").read_text().splitlines()]
    synthetic = [p for p in pages if p["source"] == "mw-synthetic"]
    human = [p for p in pages if p["source"] == "mw-human"]
    rows = []  # (title, [images]) ; the last image of each row is the overlay
    for page in synthetic[:40]:
        result = scan_document(load_document(page["path"]), layers=("reuse",), workers=1)
        groups = [e for e in result.layers["reuse"].evidence if e["kind"] == "reused" and len(e["members"]) >= 3]
        if len(groups) >= 2:
            for item in groups[:2]:
                crops = [_to_height(_crop(result.document.pages[m["page"]].pixels, m["box"], 8), 120) for m in item["members"][:3]]
                d = item["pairs"][0]["displacement"]
                rows.append((f"Generated (MathWriting synthetic): {len(item['members'])} copies; overlay d = {d:.3f}",
                             crops + [pair_overlay(result, item["pairs"][0], height=120)]))
            break
    # Real pages: every complex piece's closest look-alike; show the median case and the closest case.
    found = []
    for page in human[:30]:
        doc = load_document(page["path"])
        pieces, inks = collect(doc)
        best = {}
        for a, b, _ in candidate_pairs(pieces):
            r = align(pieces[a].comp, pieces[b].comp, pieces[b].stroke_width, early_exit=0.9)
            if r.ok and np.isfinite(r.displacement):
                for i in (a, b):
                    if i not in best or r.displacement < best[i][0]:
                        best[i] = (r.displacement, a, b, r.matrix)
        for d, a, b, m in best.values():
            found.append((d, doc, inks, pieces, a, b, m))
    found.sort(key=lambda f: f[0])
    picks = [("closest", found[0]), ("median", found[len(found) // 2])] if found else []
    for name, (d, doc, inks, pieces, a, b, m) in picks:
        holder = type("H", (), {})()
        holder.inks, holder.document = inks, doc
        holder.reuse = type("F", (), {"pieces": pieces})()
        crops = [_to_height(_crop(doc.pages[pieces[i].comp.page].pixels, pieces[i].comp.box, 8), 120) for i in (a, b)]
        rows.append((f"Real (MathWriting human), the {name} pair of {len(found):,} letters on 30 pages; overlay d = {d:.3f}",
                     crops + [pair_overlay(holder, {"a": a, "b": b, "matrix": np.asarray(m).tolist()}, height=120)]))
    if not rows:
        return
    columns = max(len(images) for _, images in rows)
    fig, axes = plt.subplots(len(rows), columns, figsize=(7.0, 1.25 * len(rows)),
                             gridspec_kw={"width_ratios": [1] * columns})
    axes = np.atleast_2d(axes)
    for r, (title, images) in enumerate(rows):
        for c in range(columns):
            ax = axes[r, c]
            ax.axis("off")
            if c < len(images) - 1:
                ax.imshow(images[c])
            elif c == columns - 1:
                ax.imshow(images[-1])
                ax.set_title("overlay", fontsize=7)
        axes[r, 0].set_title(title, fontsize=7.5, loc="left", x=0.0)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "evidence.pdf", dpi=300)
    fig.savefig(HERE / "figures" / "evidence.png", dpi=150)
    print("wrote figures/evidence.pdf with", len(rows), "rows")


if __name__ == "__main__":
    main()
