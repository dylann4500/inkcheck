"""Trains the printed-type model (inkcheck/printed.py) and tests it.

    python -I -m inkcheck.bench.train_printed [--workers 10] [--type-pages 240] [--hand-files 140] [--save]

Typeset examples: lines of worksheet text set in the Mac's ordinary text
faces (handwriting-style faces left out), 8–16 pt at 288 dpi, black, grey or
navy, each page passed through a random degradation. Handwriting examples:
Scribe pages from the manifest (every hand) and the real iPad pages (their
PDF text layer marks the printed parts, which become typeset examples).

Reports held-out accuracy: typeset recall with whole font families held out,
and how often handwriting is called type, by source and hand, with the
hands held out. The model is written to inkcheck/models/printed.npz only
with --save; it is trained on private handwriting, so git ignores it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier

from ..degrade import degrade
from ..ink import components, extract_ink
from ..load import load_document
from ..printed import save_model, vector
from ..reuse import complexity
from ..shapes import describe

ROOT = Path(__file__).resolve().parents[2]
FONT_DIRS = [Path("/System/Library/Fonts"), Path("/System/Library/Fonts/Supplemental"), Path("/Library/Fonts")]
try:  # Computer Modern (what LaTeX worksheets are set in), DejaVu and STIX, as matplotlib ships them
    import matplotlib

    FONT_DIRS.append(Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf")
except ImportError:
    pass
TEXT_FACES = re.compile(r"^(Arial|Times|Georgia|Courier|Verdana|Tahoma|Trebuchet|Helvetica|Avenir|Menlo|Monaco|Palatino|"
                        r"Baskerville|BigCaslon|Bodoni 72|Charter|Cochin|Didot|Futura|GillSans|Hoefler Text|Iowan|Athelas|"
                        r"Marion|Andale|AmericanTypewriter|Microsoft Sans|NewYork|LucidaGrande|Geneva|Optima|Rockwell|"
                        r"STIX|Seravek|SFNS|Galvji|DIN|Impact|Superclarendon|Skia|Kefa|PTSans|PTSerif|PT Sans|PT Serif|"
                        r"cmr10|cmb10|cmti10|cmss10|cmtt10|cmmi10|DejaVuSans|DejaVuSerif)", re.I)
HAND_STYLE = re.compile(r"STIXSiz|STIXNonUni|Display|Chancery|Bradley|Brush|Chalk|Comic|Corsiva|Herculanum|Luminari|Marker|Noteworthy|Zapfino|Snell|"
                        r"Savoye|SignPainter|Trattatello|Papyrus|Party|Phosphate|Ornament|Braille|Emoji|Symbol|Wingding|Webding", re.I)
CONDITIONS = ("none", "none", "jpeg", "print-scan", "photo", "downscale")


def text_fonts() -> list[tuple[str, int, str]]:
    out = []
    for folder in FONT_DIRS:
        for path in sorted(folder.glob("*.tt[cf]")) if folder.exists() else []:
            name = path.stem
            if not TEXT_FACES.match(name) or HAND_STYLE.search(name):
                continue
            for index in range(4 if path.suffix == ".ttc" else 1):
                try:
                    face = ImageFont.truetype(str(path), 40, index=index)
                    # Some faces crash the renderer ("stack overflow"); try one line first.
                    ImageDraw.Draw(Image.new("L", (900, 80))).text((5, 5), "Solve 3x + 2 = 11; the cell's DNA.", font=face, fill=255)
                except Exception:
                    break
                out.append((str(path), index, re.split(r"[ -]", name)[0]))
    return out


def corpus() -> list[str]:
    lines = []
    for path in sorted((ROOT / "data" / "scribe" / "worksheets").glob("*.json")):
        try:
            ws = json.loads(path.read_text()).get("worksheet", {})
        except Exception:
            continue
        for problem in ws.get("problems", []):
            for text in [problem.get("prompt", ""), problem.get("answer", "")] + list(problem.get("work", [])):
                text = re.sub(r"[^\x20-\x7e]", " ", str(text)).strip()
                if 8 <= len(text) <= 90:
                    lines.append(text)
    return lines or ["Solve each equation and show your work.", "The cell membrane controls what enters the cell.",
                     "Find the derivative of f(x) = 3x^2 + 2x - 5.", "Explain why the reaction is exothermic."]


def typeset_page(args) -> list[tuple]:
    seed, font, lines = args
    rng = np.random.default_rng(seed)
    path, index, family = font
    size = int(rng.integers(32, 66))
    face = ImageFont.truetype(path, size, index=index)
    page = Image.new("RGB", (2448, 1400), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    colour = [(0, 0, 0), (40, 40, 40), (70, 70, 70), (25, 35, 90)][int(rng.integers(0, 4))]
    y = 60
    for line in rng.choice(lines, size=min(len(lines), 12), replace=False):
        try:
            draw.text((80, y), str(line), font=face, fill=colour)
        except Exception:  # a glyph this face cannot draw
            continue
        y += int(size * 1.6)
        if y > 1300:
            break
    pixels = np.asarray(page)
    condition = CONDITIONS[int(rng.integers(0, len(CONDITIONS)))]
    dpi = 288.0
    if condition != "none":
        pixels, dpi = degrade(pixels, dpi, condition, seed)
    return [(row, 1, f"font:{family}", f"type:{family}", condition) for row in _pieces(pixels, dpi, rng)]


def _pieces(pixels, dpi, rng, text_boxes=(), want=None, cap=120):
    ink = extract_ink(pixels, dpi)
    rows = []
    for c in components(ink, 0, text_boxes):
        if want is not None and c.printed != want:
            continue
        info = describe(c, ink.stroke_width)
        if info.length_sw < 8 or complexity(info) < 6:
            continue
        rows.append(vector(c, ink.stroke_width, ink.letter_height))
    rng.shuffle(rows)
    return rows[:cap]


def handwriting_file(args) -> list[tuple]:
    row, seed = args
    rng = np.random.default_rng(seed)
    out = []
    doc = load_document(row["path"])
    if doc.error:
        return out
    for page in doc.pages[: (3 if row["label"] == "real" else 1)]:
        if row["source"] == "ipad":
            # Geometry-preserving conditions only, so the PDF's text boxes still mark the printed parts.
            condition = ["none", "jpeg"][int(rng.integers(0, 2))]
            pixels, dpi = (page.pixels, page.dpi) if condition == "none" else degrade(page.pixels, page.dpi, condition, seed)
            for want, label in ((False, 0), (True, 1)):
                for v in _pieces(pixels, dpi, rng, page.facts.text_boxes, want=want):
                    out.append((v, label, f"writer:{row.get('writer')}", "ipad-print" if label else "ipad", condition))
        else:
            condition = CONDITIONS[int(rng.integers(0, len(CONDITIONS)))]
            pixels, dpi = (page.pixels, page.dpi) if condition == "none" else degrade(page.pixels, page.dpi, condition, seed)
            kind = f"scribe:{row.get('hand')}"
            for v in _pieces(pixels, dpi, rng):
                out.append((v, 0, f"hand:{row.get('hand')}", kind, condition))
    return out


def build(args) -> dict:
    fonts = text_fonts()
    lines = corpus()
    print(f"{len(fonts)} text faces, {len(lines)} lines of text", flush=True)
    rnd = random.Random(11)
    type_jobs = [(1000 + i, fonts[rnd.randrange(len(fonts))], lines) for i in range(args.type_pages)]
    manifest = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    scribe = [r for r in manifest if r["label"] == "scribe"]
    rnd.shuffle(scribe)
    hand_jobs = [(r, 5000 + i) for i, r in enumerate(scribe[: args.hand_files] + [r for r in manifest if r["source"] == "ipad"])]
    items = []
    with ProcessPoolExecutor(args.workers) as pool:
        for part in pool.map(typeset_page, type_jobs, chunksize=4):
            items.extend(part)
        for part in pool.map(handwriting_file, hand_jobs, chunksize=1):
            items.extend(part)
    data = {"X": np.stack([i[0] for i in items]), "y": np.array([i[1] for i in items]),
            "group": np.array([i[2] for i in items]), "kind": np.array([i[3] for i in items]),
            "condition": np.array([i[4] for i in items])}
    np.savez_compressed(ROOT / "data" / "printed-features.npz", **data)
    return data


def fit(X, y):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=0)
    model.fit((X - mean) / scale, y)
    return model, mean, scale


def evaluate(data) -> list[str]:
    X, y, group, kind = data["X"], data["y"], data["group"], data["kind"]
    p = np.zeros(len(y))
    for train, test in GroupKFold(n_splits=5).split(X, y, group):
        model, mean, scale = fit(X[train], y[train])
        p[test] = model.predict_proba((X[test] - mean) / scale)[:, 1]
    lines = ["| Pieces (groups held out) | Count | Called printed (p > 0.5) |", "|---|---:|---:|"]
    for k in sorted(set(kind), key=lambda k: (not k.startswith("type"), k)):
        sel = kind == k
        lines.append(f"| {k} | {sel.sum()} | {np.mean(p[sel] > 0.5) * 100:.1f}% |")
    typeset, hand = y == 1, (y == 0)
    lines += ["", f"Printed type recalled: {np.mean(p[typeset] > 0.5) * 100:.1f}% of {typeset.sum()}; "
              f"handwriting called printed: {np.mean(p[hand] > 0.5) * 100:.2f}% of {hand.sum()}."]
    for c in sorted(set(data["condition"])):
        sel = data["condition"] == c
        lines.append(f"- {c}: type recalled {np.mean(p[sel & typeset] > 0.5) * 100:.1f}%, handwriting called printed "
                     f"{np.mean(p[sel & hand] > 0.5) * 100:.2f}%")
    return lines


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the printed-type model.")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--type-pages", type=int, default=240)
    parser.add_argument("--hand-files", type=int, default=140)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)
    cache = ROOT / "data" / "printed-features.npz"
    data = dict(np.load(cache, allow_pickle=False)) if cache.exists() and not args.rebuild else build(args)
    lines = evaluate(data)
    out = ROOT / "runs" / "printed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.save:
        model, mean, scale = fit(data["X"], data["y"])
        path = Path(__file__).resolve().parents[1] / "models" / "printed.npz"
        path.parent.mkdir(exist_ok=True)
        save_model(path, mean, scale, model)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
