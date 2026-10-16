"""The trained-model approach, for comparison with the twin test.

A small convolutional network looks at ink patches and learns "generated"
against "real" from labelled pages. Pages are normalised first (darkness
against the paper, ruling removed, rescaled so the pen is 3 px wide), so it
learns from the ink rather than the paper; it is trained with degraded
copies mixed in; a page's score is the mean of its patches' probabilities,
and a document's the mean of its pages'.

    python -I -m inkcheck.bench.learned extract [--workers 12]      # data/ml/: patches for every page and condition
    python -I -m inkcheck.bench.learned run [--protocols all]       # runs/learned/<protocol>.json
    python -I -m inkcheck.bench.learned report                      # tables beside the twin test's

Protocols (each trains its own model; the test set is never seen in training):

- random: documents split 80/20 at random, every source on both sides.
- holdout-real:<source>: one real source (cvl, ipad, mw-human) left out of
  training entirely, then tested: does the model false-alarm on real
  handwriting from somewhere new?
- holdout-gen:<source>: one generator left out (scribe, or mw-synthetic, the
  MathWriting glyph-stitcher): does the model catch a generator it never saw?
- clean-only: trained on undegraded pages only, tested on every degradation.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from ..degrade import degrade_document
from ..ink import extract_ink
from ..load import load_document

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml"
OUT = ROOT / "runs" / "learned"
CONDITIONS = ("none", "jpeg", "downscale", "print-scan", "photo")
PATCH = 64
PEN_PX = 3.0
PER_PAGE = 32
MAX_PAGES = {"generated": 3, "export": 3, "ipad": 4}
POSITIVE = {"scribe", "synthetic"}


# ---------------------------------------------------------------------------
# Patches


def page_patches(pixels: np.ndarray, dpi: float, rng, n: int = PER_PAGE) -> np.ndarray:
    ink = extract_ink(pixels, dpi)
    f = PEN_PX / max(ink.stroke_width, 0.5)
    dark = cv2.resize(ink.darkness, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC).clip(0, 1)
    mask = dark > 0.3
    ys, xs = np.nonzero(mask)
    h = PATCH // 2
    keep = (ys >= h) & (xs >= h) & (ys < dark.shape[0] - h) & (xs < dark.shape[1] - h)
    ys, xs = ys[keep], xs[keep]
    out = []
    if len(ys):
        for i in rng.choice(len(ys), size=min(len(ys), n * 4), replace=False):
            patch = dark[ys[i] - h : ys[i] + h, xs[i] - h : xs[i] + h]
            if (patch > 0.3).mean() >= 0.03:
                out.append((patch * 255).round().astype(np.uint8))
            if len(out) == n:
                break
    return np.stack(out) if out else np.zeros((0, PATCH, PATCH), np.uint8)


def extract_file(args) -> list[dict]:
    row, seed = args
    doc = load_document(row["path"])
    if doc.error:
        return []
    doc.pages = doc.pages[: MAX_PAGES.get(row["source"], 1)]
    items = []
    for condition in CONDITIONS:
        d = degrade_document(doc, condition, seed=seed)
        for page in d.pages:
            rng = np.random.default_rng(seed * 31 + page.index * 7 + CONDITIONS.index(condition))
            patches = page_patches(page.pixels, page.dpi, rng)
            if len(patches) >= 4:
                items.append({"doc": row["id"], "page": page.index, "condition": condition, "label": row["label"],
                              "source": row["source"], "group": row.get("hand") if row["label"] == "scribe" else row.get("writer"),
                              "patches": patches})
    return items


def _extract_one(args) -> str:
    """One file's patches, saved under data/ml/parts/ (so an interrupted run resumes)."""
    row, seed, part = args
    items = extract_file((row, seed))
    np.savez(part, patches=np.concatenate([i["patches"] for i in items]) if items else np.zeros((0, PATCH, PATCH), np.uint8),
             meta=json.dumps([{k: v for k, v in i.items() if k != "patches"} | {"count": len(i["patches"])} for i in items]))
    return str(part)


CVL_SAMPLE = 400  # the same fixed sample the benchmark degrades (bench.run --sample, crc32 order)


def _in_scope(rows: list[dict]) -> set[str]:
    """Every file, except CVL: its fixed 400-page sample."""
    import zlib

    cvl = sorted((r for r in rows if r["source"] == "cvl"), key=lambda r: zlib.crc32(r["id"].encode()))[:CVL_SAMPLE]
    return {r["id"] for r in rows if r["source"] != "cvl"} | {r["id"] for r in cvl}


def extract(workers: int) -> None:
    rows = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    keep = _in_scope(rows)
    parts = DATA / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = [(r, i, parts / f"{i:05d}.npz") for i, r in enumerate(rows) if r["id"] in keep]
    todo = [j for j in jobs if not j[2].exists()]
    started = time.time()
    print(f"{len(todo)} of {len(jobs)} files to extract", flush=True)
    with ProcessPoolExecutor(workers) as pool:
        for n, _ in enumerate(pool.map(_extract_one, todo, chunksize=1), 1):
            if n % 100 == 0:
                print(f"  {n}/{len(todo)} files, {time.time() - started:.0f}s", flush=True)
    meta, chunks, offset = [], [], 0
    for _, _, part in jobs:
        with np.load(part, allow_pickle=False) as data:
            patches = data["patches"]
            for item in json.loads(str(data["meta"])):
                item.update(start=offset)
                offset += item["count"]
                meta.append(item)
            chunks.append(patches)
    np.save(DATA / "patches.npy", np.concatenate(chunks))
    (DATA / "pages.json").write_text(json.dumps(meta))
    print(f"{len(meta)} page-conditions, {offset} patches -> {DATA}")


# ---------------------------------------------------------------------------
# Model


def make_model(width: float = 1.0):
    """The network; `width` scales every layer's channels (weights scale with its square:
    0.5 ≈ 0.15 M, 1 ≈ 0.6 M, 2 ≈ 2.4 M)."""
    import torch.nn as nn

    c = [max(4, int(round(k * width))) for k in (32, 64, 128, 256)]

    def block(i, o):
        return [nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True)]

    return nn.Sequential(*block(1, c[0]), *block(c[0], c[0]), nn.MaxPool2d(2), *block(c[0], c[1]), *block(c[1], c[1]), nn.MaxPool2d(2),
                         *block(c[1], c[2]), *block(c[2], c[2]), nn.MaxPool2d(2), *block(c[2], c[3]), nn.AdaptiveAvgPool2d(1),
                         nn.Flatten(), nn.Dropout(0.3), nn.Linear(c[3], 1))


def _device():
    import torch

    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train(patches: np.ndarray, pages: list[dict], epochs: int = 6, seed: int = 0, width: float = 1.0):
    """A model from these pages' patches; classes and sources balanced by sampling weights."""
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx, labels, weights = [], [], []
    by_source: dict[str, int] = {}
    for p in pages:
        by_source[p["source"]] = by_source.get(p["source"], 0) + p["count"]
    by_class: dict[int, int] = {}
    for p in pages:
        y = int(p["label"] in POSITIVE)
        by_class[y] = by_class.get(y, 0) + p["count"]
    sources_in_class = {y: len({p["source"] for p in pages if int(p["label"] in POSITIVE) == y}) for y in by_class}
    for p in pages:
        y = int(p["label"] in POSITIVE)
        w = 1.0 / (by_source[p["source"]] * sources_in_class[y])  # each class half, each source an equal share of it
        idx.extend(range(p["start"], p["start"] + p["count"]))
        labels.extend([y] * p["count"])
        weights.extend([w] * p["count"])
    idx, labels = np.array(idx), np.array(labels, np.float32)
    weights = np.array(weights) / np.sum(weights)
    device = _device()
    model = make_model(width).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = max(1, len(idx) // 256)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=2e-3, total_steps=epochs * steps_per_epoch)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for _ in range(steps_per_epoch):
            pick = rng.choice(len(idx), size=256, p=weights)
            batch = patches[idx[pick]].astype(np.float32) / 255
            if rng.random() < 0.5:
                batch = batch[:, :, ::-1]
            batch = batch * rng.uniform(0.8, 1.2, (len(batch), 1, 1)).astype(np.float32)
            x = torch.from_numpy(np.ascontiguousarray(batch[:, None])).to(device)
            y = torch.from_numpy(labels[pick]).to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss.detach())
        print(f"    epoch {epoch + 1}/{epochs} loss {total / steps_per_epoch:.4f}", flush=True)
    return model


def score_pages(model, patches: np.ndarray, pages: list[dict]) -> np.ndarray:
    import torch

    device = _device()
    model.eval()
    out = []
    with torch.no_grad():
        for p in pages:
            x = torch.from_numpy(patches[p["start"] : p["start"] + p["count"]].astype(np.float32)[:, None] / 255).to(device)
            out.append(float(torch.sigmoid(model(x).squeeze(1)).mean()))
    return np.array(out)


# ---------------------------------------------------------------------------
# Protocols


def _split_docs(pages, seed=0):
    docs = sorted({p["doc"] for p in pages})
    rng = np.random.default_rng(seed)
    test = set(rng.choice(docs, size=len(docs) // 5, replace=False).tolist())
    return test


def protocols(pages: list[dict]) -> dict[str, tuple]:
    """name → (train page filter, test page filter)."""
    test_docs = _split_docs(pages)
    is_test = lambda p: p["doc"] in test_docs
    out = {
        "random": (lambda p: not is_test(p), is_test),
        "clean-only": (lambda p: not is_test(p) and p["condition"] == "none", is_test),
    }
    for source in ("cvl", "ipad", "mw-human"):
        out[f"holdout-real:{source}"] = (lambda p, s=source: not is_test(p) and p["source"] != s,
                                         lambda p, s=source: is_test(p) or p["source"] == s)
    out["holdout-gen:scribe"] = (lambda p: not is_test(p) and p["label"] != "scribe",
                                 lambda p: (is_test(p) and p["label"] != "synthetic") or p["label"] == "scribe")
    out["holdout-gen:mw-synthetic"] = (lambda p: not is_test(p) and p["label"] != "synthetic",
                                       lambda p: (is_test(p) and p["label"] != "scribe") or p["label"] == "synthetic")
    return out


def run(names: list[str], epochs: int, width: float = 1.0) -> None:
    patches = np.load(DATA / "patches.npy", mmap_mode="r")
    pages = json.loads((DATA / "pages.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    table = protocols(pages)
    for name in names:
        train_filter, test_filter = table[name]
        train_pages = [p for p in pages if train_filter(p)]
        test_pages = [p for p in pages if test_filter(p)]
        print(f"{name}: training on {len(train_pages)} page-conditions, testing on {len(test_pages)}", flush=True)
        started = time.time()
        model = train(np.asarray(patches), train_pages, epochs=epochs, width=width)
        scores = score_pages(model, np.asarray(patches), test_pages)
        docs: dict[tuple, list] = {}
        for p, s in zip(test_pages, scores):
            docs.setdefault((p["doc"], p["condition"]), []).append((s, p))
        rows = [{"doc": d, "condition": c, "label": items[0][1]["label"], "source": items[0][1]["source"],
                 "score": float(np.mean([s for s, _ in items]))} for (d, c), items in docs.items()]
        target = OUT if width == 1.0 else OUT / "sweep"
        target.mkdir(parents=True, exist_ok=True)
        stem = name.replace(":", "_") if width == 1.0 else f"w{width:g}_{name.replace(':', '_')}"
        params = sum(p.numel() for p in model.parameters())
        (target / f"{stem}.json").write_text(json.dumps({"protocol": name, "width": width, "parameters": params,
                                                         "seconds": round(time.time() - started), "train_pages": len(train_pages), "docs": rows}))
        print(f"  done in {time.time() - started:.0f}s", flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="The trained-model approach.")
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--workers", type=int, default=12)
    r = sub.add_parser("run")
    r.add_argument("--protocols", default="all")
    r.add_argument("--epochs", type=int, default=6)
    r.add_argument("--width", type=float, default=1.0, help="channel multiplier (weights scale with its square)")
    sub.add_parser("report")
    args = parser.parse_args(argv)
    if args.command == "extract":
        extract(args.workers)
    elif args.command == "run":
        pages = json.loads((DATA / "pages.json").read_text())
        names = list(protocols(pages)) if args.protocols == "all" else args.protocols.split(",")
        run(names, args.epochs, args.width)
    else:
        from .compare import main as report

        report()


if __name__ == "__main__":
    main()
