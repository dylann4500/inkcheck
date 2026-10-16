"""Runs every check on every manifest file, under each condition.

    python -I -m inkcheck.bench.run --name main [--conditions none,jpeg,print-scan,photo,downscale]
        [--sources generated,export,ipad,cvl] [--limit N] [--workers 10] [--max-pages 6]

Writes runs/<name>/results.jsonl, one line per (file, condition), and skips
lines already there, so a run can be stopped and resumed. Twins are found
once at the loosest threshold in THRESHOLDS and re-sorted at each tighter
one, so calibration can pick the threshold afterwards.
"""

from __future__ import annotations

import argparse
import json
import zlib
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import binom

from ..degrade import degrade_document
from ..load import load_document
from ..provenance import assess_provenance
from ..reuse import at_threshold, extra_copies, find_twins, layer_from, load_calibration
from ..verdict import judge

ROOT = Path(__file__).resolve().parents[2]
THRESHOLDS = (0.06, 0.07, 0.08, 0.09, 0.10)
DEFAULT_CONDITIONS = ("none", "jpeg", "print-scan", "photo", "downscale")


def counts_at(findings, threshold: float, null_rate: float) -> dict:
    twins = at_threshold(findings, threshold).twins
    reused = {i for m in twins if m.kind == "reused" for i in (m.a, m.b)}
    printed = {i for m in twins if m.kind == "printed" for i in (m.a, m.b)}
    blocks = {i for m in twins if m.kind == "block" for i in (m.a, m.b)}
    n, copies = findings.complex_count, extra_copies(twins)
    p = float(binom.sf(copies - 1, n, null_rate)) if n and copies else 1.0
    return {"dup": len(reused), "copies": copies, "printed": len(printed), "block_pieces": len(blocks), "p": p}


def evaluate(row: dict, condition: str, max_pages: int) -> dict:
    started = time.time()
    out = {"key": f"{row['id']}|{condition}", "condition": condition, **{k: v for k, v in row.items() if k != "path"}}
    try:
        doc = load_document(row["path"])
        if doc.error:
            out["error"] = doc.error
            return out
        doc.pages = doc.pages[:max_pages]
        doc = degrade_document(doc, condition, seed=zlib.crc32(row["id"].encode()) % 10_000)
        cal = load_calibration()
        provenance = assess_provenance(doc)
        findings, inks = find_twins(doc, workers=1, threshold=max(THRESHOLDS))
        layer = layer_from(at_threshold(findings, cal["twin_displacement"]), inks)
        verdict = judge({"provenance": provenance, "reuse": layer})
        out.update(
            pages=len(doc.pages),
            copies=layer.data.get("extra_copies"),
            provenance={"source": provenance.data.get("source"), "signature": provenance.data.get("signature"), "score": provenance.score},
            reuse={"complex": findings.complex_count, "pen": float(np.median([i.stroke_width for i in inks])) if inks else 0.0,
                   "too_little": layer.data.get("too_little_reason"), "score": layer.score, "dup": layer.data.get("duplicated"),
                   "blocks": layer.data.get("copied_blocks"), "printed": layer.data.get("printed"), "p": layer.data.get("p_value"),
                   "by_threshold": {f"{t:.2f}": counts_at(findings, t, cal["null_rate"]) for t in THRESHOLDS}},
            verdict=verdict.label,
        )
    except Exception as exc:  # one bad file must not stop a long run
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-2000:]
    out["seconds"] = round(time.time() - started, 2)
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run inkcheck over the manifest.")
    parser.add_argument("--name", default="main")
    parser.add_argument("--manifest", default=str(ROOT / "data" / "manifest.jsonl"))
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--sources", default="")
    parser.add_argument("--limit", type=int, default=0, help="at most this many files per source")
    parser.add_argument("--sample", type=int, default=0, help="a fixed, spread sample of this many files per source (crc32 order)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 3))
    parser.add_argument("--max-pages", type=int, default=6)
    args = parser.parse_args(argv)

    rows = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if args.sources:
        wanted = set(args.sources.split(","))
        rows = [r for r in rows if r["source"] in wanted]
    if args.sample:
        by_source: dict[str, list] = {}
        for r in rows:
            by_source.setdefault(r["source"], []).append(r)
        rows = [r for group in by_source.values() for r in sorted(group, key=lambda r: zlib.crc32(r["id"].encode()))[: args.sample]]
    if args.limit:
        kept, seen = [], {}
        for r in rows:
            seen[r["source"]] = seen.get(r["source"], 0) + 1
            if seen[r["source"]] <= args.limit:
                kept.append(r)
        rows = kept
    out_dir = ROOT / "runs" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    results = out_dir / "results.jsonl"
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["key"])
    tasks = [(r, c) for r in rows for c in args.conditions.split(",") if f"{r['id']}|{c}" not in done]
    print(f"{len(tasks)} to run ({len(done)} already done), {args.workers} workers -> {results}", flush=True)
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool, results.open("a") as sink:
        futures = [pool.submit(evaluate, r, c, args.max_pages) for r, c in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            row = future.result()
            sink.write(json.dumps(row) + "\n")
            sink.flush()
            if count % 25 == 0 or count == len(futures):
                print(f"  {count}/{len(futures)} done, {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
