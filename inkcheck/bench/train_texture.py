"""Trains and tests the ink-texture model (inkcheck/texture.py).

    python -I -m inkcheck.bench.train_texture [--workers 10] [--cvl 150] [--save]

1. Edge-patch features for up to three pages of each manifest file, under
   each of none / jpeg / print-scan / photo, cached in
   data/texture-features.npz (CVL is sampled: one page from each of
   `--cvl` writers).
2. Cross-validation by group (the hand for Scribe pages, the writer for
   real ones): page-level ROC AUC per condition on held-out groups.
3. Transfer: trained on undegraded pages only, tested on each degraded
   condition; and trained without each real source, tested on it.
4. With --save, a model trained on everything goes to
   inkcheck/models/texture.npz, its notes naming the numbers above.
"""

from __future__ import annotations

import argparse
import json
import zlib
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..degrade import degrade_document
from ..ink import extract_ink
from ..load import load_document
from ..texture import page_features, save_model

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("none", "jpeg", "print-scan", "photo")
CACHE = ROOT / "data" / "texture-features.npz"


def _group(row: dict) -> str:
    if row["label"] == "scribe":
        return f"hand:{row.get('hand')}"
    return f"writer:{row.get('writer')}"


def extract(row: dict) -> list[tuple]:
    out = []
    doc = load_document(row["path"])
    if doc.error:
        return out
    doc.pages = doc.pages[:3]
    for condition in CONDITIONS:
        d = degrade_document(doc, condition, seed=zlib.crc32(row["id"].encode()) % 10_000)
        for page in d.pages:
            ink = extract_ink(page.pixels, page.dpi)
            if ink.mask.sum() < 500:
                continue
            rows = page_features(ink, n=120, seed=page.index)
            if len(rows):
                out.append((row["id"], row["label"], row["source"], _group(row), condition, page.index, rows))
    return out


def build(rows: list[dict], workers: int) -> dict:
    items = []
    with ProcessPoolExecutor(workers) as pool:
        for done, part in enumerate(pool.map(extract, rows, chunksize=1), 1):
            items.extend(part)
            if done % 20 == 0:
                print(f"  features: {done}/{len(rows)} files", flush=True)
    sizes = [len(i[6]) for i in items]
    data = {
        "X": np.concatenate([i[6] for i in items]),
        "page": np.repeat(np.arange(len(items)), sizes),
        "id": np.array([i[0] for i in items]),
        "label": np.array([1 if i[1] == "scribe" else 0 for i in items]),
        "source": np.array([i[2] for i in items]),
        "group": np.array([i[3] for i in items]),
        "condition": np.array([i[4] for i in items]),
    }
    np.savez_compressed(CACHE, **data)
    return data


def _fit(X, y):
    mean, scale = X.mean(0), X.std(0) + 1e-6
    model = LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced")
    model.fit((X - mean) / scale, y)
    return model, mean, scale


def _page_scores(model, mean, scale, data, pages):
    out = []
    for p in pages:
        rows = data["X"][data["page"] == p]
        out.append(float(model.predict_proba((rows - mean) / scale)[:, 1].mean()))
    return np.array(out)


def evaluate(data: dict) -> tuple[list[str], dict]:
    n_pages = len(data["label"])
    pages = np.arange(n_pages)
    lines, summary = [], {}
    # 1. Cross-validation by group.
    folds = GroupKFold(n_splits=5)
    scores = np.zeros(n_pages)
    for train, test in folds.split(pages, data["label"], data["group"]):
        mask = np.isin(data["page"], train)
        model, mean, scale = _fit(data["X"][mask], data["label"][data["page"][mask]])
        scores[test] = _page_scores(model, mean, scale, data, test)
    lines += ["| Held-out groups, condition | Pages (Scribe / real) | AUC |", "|---|---:|---:|"]
    for c in CONDITIONS:
        sel = data["condition"] == c
        y = data["label"][sel]
        if len(set(y)) == 2:
            auc = roc_auc_score(y, scores[sel])
            summary[f"cv:{c}"] = auc
            lines.append(f"| {c} | {int(y.sum())} / {int((1 - y).sum())} | {auc:.3f} |")
    # 2. Trained on undegraded pages only.
    lines += ["", "| Trained on undegraded only, tested on | AUC |", "|---|---:|"]
    clean = pages[data["condition"] == "none"]
    mask = np.isin(data["page"], clean)
    model, mean, scale = _fit(data["X"][mask], data["label"][data["page"][mask]])
    for c in CONDITIONS[1:]:
        test = pages[data["condition"] == c]
        y = data["label"][test]
        if len(set(y)) == 2:
            auc = roc_auc_score(y, _page_scores(model, mean, scale, data, test))
            summary[f"clean->{c}"] = auc
            lines.append(f"| {c} | {auc:.3f} |")
    # 3. Leave one real source out.
    lines += ["", "| Real source left out of training | Its pages called Scribe (score > 0.5), undegraded |", "|---|---:|"]
    for source in sorted({s for s, y in zip(data["source"], data["label"]) if y == 0}):
        held = pages[(data["source"] == source)]
        train = pages[(data["source"] != source)]
        mask = np.isin(data["page"], train)
        model, mean, scale = _fit(data["X"][mask], data["label"][data["page"][mask]])
        test = held[data["condition"][held] == "none"]
        s = _page_scores(model, mean, scale, data, test)
        summary[f"heldout-real:{source}"] = float(np.mean(s > 0.5)) if len(s) else None
        lines.append(f"| {source} | {np.mean(s > 0.5) * 100:.1f}% of {len(s)} |")
    return lines, summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train and test the ink-texture model.")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifest.jsonl"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--cvl", type=int, default=150, help="CVL writers to sample (one page each)")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args(argv)
    if CACHE.exists() and not args.rebuild:
        data = dict(np.load(CACHE, allow_pickle=False))
    else:
        rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
        cvl = [r for r in rows if r["source"] == "cvl"]
        by_writer: dict[str, dict] = {}
        for r in cvl:
            by_writer.setdefault(r["writer"], r)
        chosen = sorted(by_writer.values(), key=lambda r: r["writer"])
        random.Random(7).shuffle(chosen)
        rows = [r for r in rows if r["source"] != "cvl"] + chosen[: args.cvl]
        print(f"extracting features from {len(rows)} files", flush=True)
        data = build(rows, args.workers)
    lines, summary = evaluate(data)
    report = "\n".join(lines) + "\n"
    out = ROOT / "runs" / "texture"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(report)
    if args.save:
        model, mean, scale = _fit(data["X"], data["label"][data["page"]])
        notes = "; ".join(f"{k} {v:.3f}" for k, v in summary.items() if v is not None)
        path = Path(__file__).resolve().parents[1] / "models" / "texture.npz"
        path.parent.mkdir(exist_ok=True)
        save_model(path, mean, scale, model.coef_[0], model.intercept_[0], notes)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
