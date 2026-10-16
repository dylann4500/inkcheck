"""The trained model against the twin test, on the same test documents.

    python -I -m inkcheck.bench.compare [--twin runs/main] [--out docs/comparison-tables.md]

For each protocol run by inkcheck.bench.learned, the CNN's document scores
are set beside the twin test's results (runs/main, from inkcheck.bench.run)
for exactly the documents and conditions both scored. Reported, per
protocol and condition:

- AUC: how well each score separates generated from real (threshold-free;
  for the twin test the score is −log10 of the binomial chance).
- False alarms: real documents called generated. The CNN at 0.5 (a balanced
  classifier's natural cut); the twin test at its verdict ("generated").
- Detection: generated documents called generated, at the same cuts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
ORDER = ["none", "jpeg", "downscale", "print-scan", "photo"]
POSITIVE = {"scribe", "synthetic"}


def _twin(run: Path) -> dict:
    out = {}
    for line in (run / "results.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("reuse") and not r.get("error"):
                p = r["reuse"]["p"]
                out[r["key"]] = {"score": -np.log10(max(1.0 if p is None else p, 1e-300)), "flag": r["verdict"] == "generated"}
    return out


def _rate(k: int, n: int, upper: bool = False) -> str:
    if n == 0:
        return "–"
    text = f"{k}/{n} ({100 * k / n:.1f}%)"
    if upper:
        text += f", ≤{100 * (beta.ppf(0.95, k + 1, n - k) if k < n else 1):.1f}%"
    return text


def _auc(y, s) -> str:
    return f"{roc_auc_score(y, s):.3f}" if len(set(y)) == 2 else "–"


def compare(protocol_file: Path, twin: dict) -> tuple[list[str], dict]:
    data = json.loads(protocol_file.read_text())
    rows = [r for r in data["docs"] if f"{r['doc']}|{r['condition']}" in twin]
    name = data["protocol"]
    lines = [f"### {name}", "", f"CNN trained on {data['train_pages']} page-conditions; "
             f"{len({r['doc'] for r in rows})} test documents scored by both.", "",
             "| Condition | AUC: twin / CNN | False alarms on real: twin | CNN | Detected: twin | CNN |",
             "|---|---|---|---|---|---|"]
    summary = {}
    for c in [c for c in ORDER if any(r["condition"] == c for r in rows)]:
        sel = [r for r in rows if r["condition"] == c]
        y = [int(r["label"] in POSITIVE) for r in sel]
        t_score = [twin[f"{r['doc']}|{c}"]["score"] for r in sel]
        t_flag = [twin[f"{r['doc']}|{c}"]["flag"] for r in sel]
        m_score = [r["score"] for r in sel]
        m_flag = [s >= 0.5 for s in m_score]
        neg = [i for i, v in enumerate(y) if v == 0]
        pos = [i for i, v in enumerate(y) if v == 1]
        lines.append(f"| {c} | {_auc(y, t_score)} / {_auc(y, m_score)} | {_rate(sum(t_flag[i] for i in neg), len(neg), True)} | "
                     f"{_rate(sum(m_flag[i] for i in neg), len(neg), True)} | {_rate(sum(t_flag[i] for i in pos), len(pos))} | "
                     f"{_rate(sum(m_flag[i] for i in pos), len(pos))} |")
        summary[c] = {"twin_fa": sum(t_flag[i] for i in neg), "cnn_fa": sum(m_flag[i] for i in neg), "neg": len(neg),
                      "twin_det": sum(t_flag[i] for i in pos), "cnn_det": sum(m_flag[i] for i in pos), "pos": len(pos)}
    # By source, all conditions pooled: where the errors are.
    lines += ["", "| Source (all conditions) | Truth | Twin test called generated | CNN called generated |", "|---|---|---|---|"]
    for source in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["source"] == source]
        truth = "generated" if sel[0]["label"] in POSITIVE else "real"
        t = sum(twin[f"{r['doc']}|{r['condition']}"]["flag"] for r in sel)
        m = sum(r["score"] >= 0.5 for r in sel)
        lines.append(f"| {source} | {truth} | {_rate(t, len(sel))} | {_rate(m, len(sel))} |")
    return lines + [""], summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--twin", default=str(ROOT / "runs" / "main"))
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    twin = _twin(Path(args.twin))
    lines = ["## The trained model against the twin test", ""]
    summaries = {}
    for path in sorted((ROOT / "runs" / "learned").glob("*.json")):
        if "docs" not in json.loads(path.read_text()):  # this script's own summary, not a protocol's results
            continue
        table, summary = compare(path, twin)
        lines += table
        summaries[path.stem] = summary
    text = "\n".join(lines) + "\n"
    (ROOT / "runs" / "learned" / "comparison.json").write_text(json.dumps(summaries, indent=2))
    if args.out:
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
