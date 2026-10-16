"""For the paper's main figure: each complex letter's stroke distance to its closest look-alike.

    python -I -m inkcheck.bench.distances [--per-source 60] [--workers 12]

Takes a fixed sample of undegraded documents from each source, finds every
complex piece's candidates as the detector does, aligns them with a looser
early exit, and keeps each piece's smallest stroke distance d. Pieces the
detector sets aside (members of twins it sorts as printed type or as a
copied block) are left out, so the real distribution is handwriting, not the
printed paragraph on a CVL form. Writes runs/distances/distances.json: lists
of d (no images) for real handwriting, Scribe, and the MathWriting
generator, the same by source, and how many pieces were set aside.
"""

from __future__ import annotations

import argparse
import json
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..load import load_document
from ..register import align, transform_parts
from ..reuse import Match, candidate_pairs, collect, load_calibration, sort_twins

ROOT = Path(__file__).resolve().parents[2]
GROUP = {"real": "real", "scribe": "scribe", "synthetic": "synthetic"}


def best_distances(row: dict) -> tuple[str, str, list[float], int]:
    doc = load_document(row["path"])
    if doc.error:
        return row["label"], row["source"], [], 0
    doc.pages = doc.pages[:2]
    pieces, _ = collect(doc)
    best = np.full(len(pieces), np.inf)
    twins = []
    threshold = load_calibration()["twin_displacement"]
    for a, b, sim in candidate_pairs(pieces):
        r = align(pieces[a].comp, pieces[b].comp, pieces[b].stroke_width, early_exit=0.85)
        if r.ok:
            best[a] = min(best[a], r.displacement)
            best[b] = min(best[b], r.displacement)
            if r.displacement < threshold:
                twins.append(Match(a, b, sim, r.displacement, r.correlation, r.matrix, transform_parts(r.matrix)))
    sort_twins(pieces, twins)
    aside = {i for m in twins if m.kind in ("printed", "block") for i in (m.a, m.b)}
    kept = [float(min(d, 9.99)) for i, d in enumerate(best) if i not in aside]
    return row["label"], row["source"], kept, len(aside)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-source", type=int, default=60)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args(argv)
    rows = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["source"], []).append(r)
    chosen = [r for group in by.values() for r in sorted(group, key=lambda r: zlib.crc32(r["id"].encode()))[: args.per_source]]
    out: dict = {k: [] for k in GROUP.values()}
    out["by_source"], out["set_aside"], out["documents"] = {}, {}, {}
    with ProcessPoolExecutor(args.workers) as pool:
        for label, source, values, aside in pool.map(best_distances, chosen, chunksize=1):
            out[GROUP[label]].extend(values)
            out["by_source"].setdefault(source, []).extend(values)
            out["set_aside"][source] = out["set_aside"].get(source, 0) + aside
            out["documents"][source] = out["documents"].get(source, 0) + 1
    target = ROOT / "runs" / "distances"
    target.mkdir(parents=True, exist_ok=True)
    (target / "distances.json").write_text(json.dumps(out))
    for k, v in [(k, out[k]) for k in GROUP.values()] + sorted(out["by_source"].items()):
        v = np.array(v)
        if len(v):
            print(f"{k:12s} pieces {len(v):6d}  d<0.08: {np.mean(v < 0.08) * 100:5.1f}%  median {np.median(v):.3f}"
                  f"  set aside: {out['set_aside'].get(k, '')}")


if __name__ == "__main__":
    main()
