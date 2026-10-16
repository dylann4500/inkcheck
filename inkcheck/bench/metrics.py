"""The numbers from a benchmark run, as markdown tables.

    python -I -m inkcheck.bench.metrics runs/main [--calibrate] [--out docs/results-tables.md]

For each condition: on real handwriting, how often each verdict came out
(a "generated" there is a false alarm; its 95% upper bound is given); on
Scribe pages, how often each verdict came out, by source, hand, messiness,
paper and export scale. Then the twin threshold: the rate of twins among
real complex letters at each threshold (with its 95% upper bound), and the
detection rate it would give. With --calibrate, the chosen threshold and a
null rate at that bound (never below 0.002) are written to calibration.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from importlib import resources
from pathlib import Path

import numpy as np
from scipy.stats import beta, binom
from sklearn.metrics import roc_auc_score

ORDER = ["none", "jpeg", "downscale", "print-scan", "photo"]
VERDICTS = ["generated", "suspicious", "no-evidence", "too-little", "error"]


def upper95(k: int, n: int) -> float:
    """One-sided 95% Clopper–Pearson upper bound on a rate seen k times in n."""
    if n == 0:
        return 1.0
    return float(beta.ppf(0.95, k + 1, n - k)) if k < n else 1.0


def load(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "results.jsonl").read_text().splitlines() if line.strip()]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def verdict_table(rows: list[dict], conditions: list[str]) -> list[str]:
    out = ["| Condition | Truth | Files | Generated | Suspicious | No evidence | Too little | Errors |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in conditions:
        for label in ("real", "scribe", "synthetic"):
            sel = [r for r in rows if r["condition"] == c and r["label"] == label]
            if not sel:
                continue
            n = len(sel)
            counts = Counter(r.get("verdict", "error") if not r.get("error") else "error" for r in sel)
            gen = counts["generated"]
            gen_text = f"{gen} ({_pct(gen / n)})" + (f", ≤{_pct(upper95(gen, n))}" if label == "real" else "")
            out.append(f"| {c} | {label} | {n} | {gen_text} | {counts['suspicious']} | {counts['no-evidence']} | "
                       f"{counts['too-little']} | {counts['error']} |")
    return out


def breakdown(rows: list[dict], field: str, conditions: list[str], label: str = "scribe") -> list[str]:
    values = sorted({str(r.get(field)) for r in rows if r["label"] == label and r.get(field) is not None})
    if not values:
        return []
    out = [f"| {field} | " + " | ".join(conditions) + " |", "|---|" + "---:|" * len(conditions)]
    for v in values:
        cells = []
        for c in conditions:
            sel = [r for r in rows if r["label"] == label and r["condition"] == c and str(r.get(field)) == v]
            cells.append(f"{sum(r.get('verdict') == 'generated' for r in sel)}/{len(sel)}" if sel else "–")
        out.append(f"| {v} | " + " | ".join(cells) + " |")
    return out


def threshold_table(rows: list[dict], conditions: list[str], min_copies: int, p_cut: float = 1e-6) -> tuple[list[str], dict]:
    taus = sorted({t for r in rows if r.get("reuse") for t in r["reuse"]["by_threshold"]})
    out = ["| Threshold | Real extra copies / complex letters | Rate (95% upper) | Real files flagged | Scribe files flagged |",
           "|---:|---:|---:|---:|---:|"]
    summary = {}
    for t in taus:
        real = [r for r in rows if r["label"] == "real" and r.get("reuse") and r["condition"] in conditions]
        scribe = [r for r in rows if r["label"] == "scribe" and r.get("reuse") and r["condition"] in conditions]
        k = sum(r["reuse"]["by_threshold"][t]["copies"] for r in real)
        n = sum(r["reuse"]["complex"] for r in real)
        rate_hi = max(0.002, upper95(k, n))

        def flagged(r):
            d = r["reuse"]["by_threshold"][t]["copies"]
            m = r["reuse"]["complex"]
            if r["reuse"].get("too_little") or not m:
                return False
            return d >= min_copies and binom.sf(d - 1, m, rate_hi) < p_cut

        fr = sum(flagged(r) for r in real)
        fs = sum(flagged(r) for r in scribe)
        summary[t] = {"k": k, "n": n, "rate_hi": rate_hi, "real_flagged": fr, "real": len(real), "scribe_flagged": fs, "scribe": len(scribe)}
        out.append(f"| {t} | {k} / {n} | {k / max(n, 1):.4f} ({rate_hi:.4f}) | {fr}/{len(real)} | {fs}/{len(scribe)} "
                   f"({_pct(fs / max(len(scribe), 1))}) |")
    return out, summary


def auc_line(rows: list[dict], condition: str) -> str:
    sel = [r for r in rows if r["condition"] == condition and r.get("reuse") and not r.get("error") and r["label"] in ("real", "scribe")]
    y = [1 if r["label"] == "scribe" else 0 for r in sel]
    if len(set(y)) < 2:
        return f"| {condition} | – |"
    s = [-np.log10(max(1.0 if r["reuse"]["p"] is None else r["reuse"]["p"], 1e-300)) for r in sel]
    return f"| {condition} | {roc_auc_score(y, s):.4f} |"


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Tables from a benchmark run.")
    parser.add_argument("run")
    parser.add_argument("--out", default="")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args(argv)
    rows = load(Path(args.run))
    conditions = [c for c in ORDER if any(r["condition"] == c for r in rows)]
    with resources.files("inkcheck").joinpath("calibration.json").open() as f:
        cal = json.load(f)
    lines = [f"Run `{args.run}`: {len(rows)} file-condition results "
             f"({len({r['id'] for r in rows if r['label'] == 'real'})} real files, "
             f"{len({r['id'] for r in rows if r['label'] == 'scribe'})} Scribe files).", "",
             "## Verdicts", ""] + verdict_table(rows, conditions)
    real_sources = sorted({r["source"] for r in rows if r["label"] == "real"})
    if real_sources:
        lines += ["", "## Real handwriting by source (false 'generated' / files)", ""]
        lines += breakdown(rows, "source", conditions, label="real")
    lines += ["", "## Scribe detection by setting (generated / files)", ""]
    for field in ("source", "hand", "messiness", "paper", "scale", "hand_kind"):
        table = breakdown(rows, field, conditions)
        if table:
            lines += table + [""]
    lines += ["## Separation (ROC AUC of the reuse test's −log10 p)", "", "| Condition | AUC |", "|---|---:|"]
    lines += [auc_line(rows, c) for c in conditions]
    table, summary = threshold_table(rows, conditions, cal["min_copies"])
    lines += ["", "## Twin threshold (all conditions pooled)", ""] + table
    real_writers = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["label"] == "real" and r.get("reuse"):
            t = f"{cal['twin_displacement']:.2f}"
            real_writers[r.get("writer")][0] += r["reuse"]["by_threshold"].get(t, {}).get("copies", 0)
            real_writers[r.get("writer")][1] += r["reuse"]["complex"]
    lines += ["", f"## Real writers at the current threshold ({cal['twin_displacement']})", "",
              "| Writer | Extra copies / complex letters |", "|---|---:|"]
    lines += [f"| {w} | {k} / {n} |" for w, (k, n) in sorted(real_writers.items(), key=lambda x: -x[1][1])[:30]]
    text = "\n".join(lines) + "\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    if args.calibrate and summary:
        clean = [t for t, s in summary.items() if s["real_flagged"] == 0]
        best = max(clean, key=lambda t: (summary[t]["scribe_flagged"], -float(t))) if clean else min(summary)
        cal["twin_displacement"] = float(best)
        cal["null_rate"] = round(summary[best]["rate_hi"], 5)
        cal["_note"] = (f"Set by inkcheck.bench.metrics --calibrate from {args.run}: {summary[best]['k']} extra copies in "
                        f"{summary[best]['n']} real complex letters at {best} (rate's 95% upper bound, never below 0.002).")
        path = Path(__file__).resolve().parents[1] / "calibration.json"
        path.write_text(json.dumps(cal, indent=2) + "\n")
        print(f"calibration.json: twin_displacement={best}, null_rate={cal['null_rate']}")


if __name__ == "__main__":
    main()
