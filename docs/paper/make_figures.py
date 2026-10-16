"""Numbers, tables and figures for the paper, from the benchmark's runs.

    .venv/bin/python -I docs/paper/make_figures.py [run]   (default: runs/final, else runs/main)

Reads runs/main (the glyph-reuse test), runs/learned (the CNN), runs/texture,
runs/printed, runs/mw-null and runs/distances, and writes numbers.tex (every
number the text quotes), table_*.tex and figures/*.pdf beside this file.
Only aggregate numbers leave runs/; the one picture of ink (the evidence
figure) is made from MathWriting's openly licensed synthetic pages.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = ROOT / "runs"
FIG = HERE / "figures"
ORDER = ["none", "jpeg", "downscale", "print-scan", "photo"]
COND_NAME = {"none": "As exported", "jpeg": "JPEG", "downscale": "Half resolution", "print-scan": "Print and scan", "photo": "Phone photo"}
POSITIVE = {"scribe", "synthetic"}
SOURCE_NAME = {"generated": "Scribe, rendered", "export": "Scribe, real exports", "ipad": "Tablet notes (real)", "cvl": "CVL scans (real)",
               "mw-human": "MathWriting human (real)", "mw-synthetic": "MathWriting synthetic"}


def pct(k: int, n: int, digits: int = 1) -> str:
    return "--" if not n else f"{100 * k / n:.{digits}f}\\%"


def pct_fine(k: int, n: int) -> str:
    """A percentage with a second decimal when it is small but not zero (false-alarm rates)."""
    return pct(k, n, 2 if n and 0 < 100 * k / n < 1 else 1)


def thousands(n: int) -> str:
    return f"{n:,}".replace(",", "{,}")


def upper95(k: int, n: int) -> float:
    from scipy.stats import beta

    return 1.0 if n == 0 or k >= n else float(beta.ppf(0.95, k + 1, n - k))


def load_twin(name: str | None = None) -> list[dict]:
    """The glyph-reuse test's results: runs/final (the final code) when it exists, else runs/main."""
    name = name or ("final" if (RUNS / "final" / "results.jsonl").exists() else "main")
    path = RUNS / name / "results.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []
    latest = {r["key"]: r for r in rows}  # a key run twice (a resumed or parallel run) counts once, its last result
    errors = [k for k, r in latest.items() if r.get("error")]
    if errors:
        print(f"warning: {len(errors)} results are errors and are left out, e.g. {errors[:3]}")
    return [r for r in latest.values() if not r.get("error")]


def flagged(row: dict, rate: float, min_copies: int, threshold: str = "0.08", p_cut: float = 1e-6) -> bool:
    """The verdict 'generated' recomputed from stored counts under another false-twin rate."""
    from scipy.stats import binom

    u = row.get("reuse") or {}
    if not u or u.get("too_little"):
        return False
    copies = u["by_threshold"][threshold]["copies"]
    n = u["complex"]
    return bool(n and copies >= min_copies and binom.sf(copies - 1, n, rate) < p_cut)


def copies_and_letters(rows: list[dict]) -> tuple[int, int]:
    """Extra copies and complex letters summed over documents (twin threshold 0.08)."""
    have = [r for r in rows if (r.get("reuse") or {}).get("by_threshold")]
    return (sum(r["reuse"]["by_threshold"]["0.08"]["copies"] for r in have), sum(r["reuse"]["complex"] for r in have))


def sensitivity(rows: list[dict]) -> tuple[str, str]:
    """The pre-set chance-copy rate against the rate measured on real handwriting.

    The rate is measured on the undegraded real documents, so that each letter counts
    once; the rate under each degradation is reported beside it as a check.
    """
    cal = json.loads((ROOT / "inkcheck" / "calibration.json").read_text())
    real = [r for r in rows if r["label"] == "real"]
    k, n = copies_and_letters([r for r in real if r["condition"] == "none"])
    measured = upper95(k, n)
    preset = cal["null_rate"]
    groups = {t: [r for r in rows if r["label"] == t and r.get("reuse")] for t in ("real", "scribe", "synthetic")}
    lines = ["\\begin{table}[t]\\centering\\caption{Sensitivity of the operating point (first row): documents called generated "
             "when one setting changes. $r$: chance copies per 1{,}000 complex letters; the measured rate is the 95\\% upper bound on "
             "the " + thousands(n) + " complex letters of the undegraded real documents. $d$: the twin threshold. All conditions "
             f"pooled: {thousands(len(groups['real']))} real, {thousands(len(groups['scribe']))} Scribe and "
             f"{thousands(len(groups['synthetic']))} MathWriting synthetic document-conditions.}}\\label{{tab:sensitivity}}",
             "\\footnotesize\\begin{tabular}{@{}lrrr@{}}\\toprule",
             "Setting & Real & Scribe & MW synth. \\\\\\midrule"]
    out = ""
    settings = (("PreSet", f"$r = {1000 * preset:.2f}$, $d < 0.08$ (used)", preset, "0.08"),
                ("Measured", f"$r = {1000 * measured:.2f}$ (measured)", measured, "0.08"),
                ("Dsix", "$d < 0.06$", preset, "0.06"), ("Dten", "$d < 0.10$", preset, "0.10"))
    for label, name, rate, threshold in settings:
        cells = []
        for truth, sel in groups.items():
            hit = sum(flagged(r, rate, cal["min_copies"], threshold) for r in sel)
            cells.append(f"{hit} ({pct_fine(hit, len(sel))})" if truth == "real" else pct(hit, len(sel)))
            out += macro(f"Sens{label}{truth.title()}", pct_fine(hit, len(sel))) + macro(f"Sens{label}{truth.title()}K", hit)
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\\end{tabular}\\end{table}")
    out += macro("MeasuredRate", f"{1000 * k / max(n, 1):.2f}") + macro("MeasuredRateUpper", f"{1000 * measured:.2f}")
    out += macro("MeasuredLetters", thousands(n)) + macro("MeasuredCopies", k)
    # The same rate under each degradation, and per real source (undegraded).
    worst = (0.0, "none")
    for c in ORDER:
        kc, nc = copies_and_letters([r for r in real if r["condition"] == c])
        out += macro(f"MeasuredRate{c.replace('-', '').title()}", f"{1000 * kc / max(nc, 1):.2f}")
        out += macro(f"MeasuredCopies{c.replace('-', '').title()}", kc) + macro(f"MeasuredLetters{c.replace('-', '').title()}", thousands(nc))
        if nc and kc / nc > worst[0]:
            worst = (kc / nc, c)
    out += macro("MeasuredWorstRate", f"{1000 * worst[0]:.2f}") + macro("MeasuredWorstCondition", COND_NAME[worst[1]].lower())
    # How clustered chance copies are: a binomial (or Poisson) rate assumes they fall independently.
    per_doc = sorted((r["reuse"]["by_threshold"]["0.08"]["copies"] for r in real if r["condition"] == "none"
                      and (r.get("reuse") or {}).get("by_threshold")), reverse=True)
    if per_doc and sum(per_doc):
        top = max(1, len(per_doc) // 100)
        out += macro("RealCopyDocs", sum(c > 0 for c in per_doc)) + macro("RealDocsNone", thousands(len(per_doc)))
        out += macro("RealCopyTopShare", f"{100 * sum(per_doc[:top]) / sum(per_doc):.0f}\\%")
        out += macro("RealCopyMax", per_doc[0]) + macro("RealCopyDispersion", f"{np.var(per_doc) / np.mean(per_doc):.1f}")
    for source in ("cvl", "ipad", "mw-human"):
        ks, ns = copies_and_letters([r for r in real if r["condition"] == "none" and r["source"] == source])
        key = re.sub(r"[^A-Za-z]", "", source.title())
        out += macro(f"Copies{key}", ks) + macro(f"Letters{key}", thousands(ns)) + macro(f"Rate{key}", f"{1000 * ks / max(ns, 1):.2f}")
        out += macro(f"Upper{key}", f"{1000 * upper95(ks, ns):.2f}")
    return "\n".join(lines) + "\n", out


def prefix_numbers() -> str:
    """The CVL false alarms before the fix for scan-broken printed letters (runs/main)."""
    rows = load_twin("main")
    cvl = [r for r in rows if r["source"] == "cvl" and r["condition"] == "none"]
    k = sum(r.get("verdict") == "generated" for r in cvl)
    return macro("CvlBeforeFixK", k) + macro("CvlBeforeFixN", f"{len(cvl):,}".replace(",", "{,}"))


def sweep() -> tuple[str, str]:
    """Capacity sweep: the same protocols at three network widths."""
    from sklearn.metrics import roc_auc_score

    rows = []
    for width in (0.5, 1.0, 2.0):
        for protocol in ("random", "holdout-real:cvl", "holdout-gen:scribe"):
            stem = protocol.replace(":", "_")
            path = (RUNS / "learned" / f"{stem}.json") if width == 1.0 else (RUNS / "learned" / "sweep" / f"w{width:g}_{stem}.json")
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            docs = data["docs"]
            params = data.get("parameters") or 582561
            if protocol == "random":
                y = [int(d["label"] in POSITIVE) for d in docs]
                value = f"{roc_auc_score(y, [d['score'] for d in docs]):.3f}"
            elif protocol == "holdout-real:cvl":
                sel = [d for d in docs if d["source"] == "cvl"]
                value = pct(sum(d['score'] >= 0.5 for d in sel), len(sel))
            else:
                sel = [d for d in docs if d["label"] == "scribe"]
                value = pct(sum(d['score'] >= 0.5 for d in sel), len(sel))
            rows.append((width, params, protocol, value, data.get("seconds")))
    if not rows:
        return "", ""
    widths = sorted({r[0] for r in rows})
    lines = ["\\begin{table}[t]\\centering\\caption{Capacity sweep: the network at three widths, each trained and tested as in "
             "Table~\\ref{tab:protocols}. Random split: AUC. Held-out CVL: CVL documents called generated. Held-out Scribe: "
             "Scribe documents caught.}\\label{tab:sweep}",
             "\\footnotesize\\begin{tabular}{@{}lrrr@{}}\\toprule", " & Random split & CVL held out & Scribe held out \\\\",
             "Weights & AUC & false alarms & caught \\\\\\midrule"]
    numbers = macro("SweepWidths", len(widths))
    names = {0.5: "Half", 1.0: "One", 2.0: "Two"}
    for w in widths:
        params = next(r[1] for r in rows if r[0] == w)
        cell = {r[2]: r[3] for r in rows if r[0] == w}
        lines.append(f"{params / 1e6:.2f} M & " + " & ".join(cell.get(p, "--") for p in ("random", "holdout-real:cvl", "holdout-gen:scribe")) + " \\\\")
        numbers += macro(f"SweepParams{names[w]}", f"{params / 1e6:.2f}~M")
        for p, short in (("random", "Auc"), ("holdout-real:cvl", "Cvl"), ("holdout-gen:scribe", "Scribe")):
            value = cell.get(p, "--")
            numbers += macro(f"Sweep{short}{names[w]}", value)
        seconds = [r[4] for r in rows if r[0] == w and r[4]]
        numbers += macro(f"SweepMinutes{names[w]}", f"{np.mean(seconds) / 60:.0f}" if seconds else "--")
    lines.append("\\bottomrule\\end{tabular}\\end{table}")
    return "\n".join(lines) + "\n", numbers


def macro(name: str, value) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


# ---------------------------------------------------------------------------


def dataset(rows: list[dict]) -> tuple[str, str]:
    manifest = [json.loads(l) for l in (ROOT / "data" / "manifest.jsonl").read_text().splitlines() if l.strip()]
    by = defaultdict(list)
    for r in manifest:
        by[r["source"]].append(r)
    pages = {"generated": sum(int(r.get("pages") or 1) for r in by["generated"]), "export": sum(int(r.get("pages") or 1) for r in by["export"])}
    writers = {"ipad": len({r["writer"] for r in by["ipad"]}), "cvl": len({r["writer"] for r in by["cvl"]})}
    hands = sorted({r.get("hand") for r in by["generated"]})
    worksheets = sorted({r.get("worksheet") for r in by["generated"]})
    lines = [
        "\\begin{table}[t]\\centering\\caption{Sources. Every document is also evaluated under four degradations.}\\label{tab:data}",
        "\\footnotesize\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{@{}llrr@{}}\\toprule",
        "Source & Truth & Documents & Writers/hands \\\\\\midrule",
        f"Scribe, real exports & generated & {len(by['export'])} & 1 writer \\\\",
        f"Scribe, rendered & generated & {len(by['generated'])} ({pages['generated']} pp.) & {len(hands)} hands \\\\",
        f"MathWriting synthetic & generated & {len(by['mw-synthetic'])} & 6{{,}}423 glyphs \\\\",
        f"Tablet notes & real & {len(by['ipad'])} & {writers['ipad']} \\\\",
        f"CVL scans & real & {thousands(len(by['cvl']))} & {writers['cvl']} \\\\",
        f"MathWriting human & real & {len(by['mw-human'])} & many (mixed) \\\\",
        "\\bottomrule\\end{tabular}\\end{table}",
    ]
    numbers = (macro("NumDocs", thousands(len(manifest))) + macro("NumRenders", pages["generated"]) +
               macro("NumExports", len(by["export"])) + macro("NumExportPages", pages["export"]) +
               macro("NumWorksheets", len(worksheets)) + macro("NumHands", len(hands)) +
               macro("NumCVLPages", thousands(len(by["cvl"]))) + macro("NumMwPages", len(by["mw-human"])) +
               macro("NumRealWriters", writers["ipad"] + writers["cvl"]))
    # Document-conditions scored by the glyph-reuse test, and its speed (one core per document).
    numbers += macro("NumDocConditions", thousands(len(rows)))
    per_page = [r["seconds"] / max(int(r.get("pages") or 1), 1) for r in rows if r.get("seconds") and r["condition"] == "none"]
    scribe_page = [r["seconds"] / max(int(r.get("pages") or 1), 1) for r in rows
                   if r.get("seconds") and r["condition"] == "none" and r["label"] == "scribe"]
    numbers += macro("SecondsPerPage", f"{np.median(per_page):.0f}" if per_page else "--")
    numbers += macro("SecondsPerScribePage", f"{np.median(scribe_page):.0f}" if scribe_page else "--")
    return "\n".join(lines) + "\n", numbers


def twin_table(rows: list[dict]) -> tuple[str, str]:
    """Verdicts of the glyph-reuse test: per condition, per truth class (and real sources)."""
    groups = [("Scribe (generated)", lambda r: r["label"] == "scribe"), ("MathWriting synth. (generated)", lambda r: r["label"] == "synthetic"),
              ("Real: tablet", lambda r: r["source"] == "ipad"), ("Real: CVL scans", lambda r: r["source"] == "cvl"),
              ("Real: MathWriting human", lambda r: r["source"] == "mw-human")]
    lines = ["\\begin{table*}[t]\\centering\\caption{Glyph-reuse test: documents called \\emph{generated}, by condition. For generated "
             "documents this is detection (documents in brackets); for real handwriting it is the false-alarm rate (95\\% upper bound "
             "in brackets). CVL pages, already scans, are not printed and scanned again, and their other degradations use a fixed "
             "sample of 400 pages.}\\label{tab:twin}",
             "\\footnotesize\\begin{tabular}{@{}l" + "r" * len(ORDER) + "@{}}\\toprule",
             "Documents & " + " & ".join(COND_NAME[c] for c in ORDER) + " \\\\\\midrule"]
    numbers = ""
    for name, test in groups:
        cells = []
        for c in ORDER:
            sel = [r for r in rows if r["condition"] == c and test(r)]
            k = sum(r.get("verdict") == "generated" for r in sel)
            if not sel:
                cells.append("--")
            elif "Real" in name:
                cells.append(f"{k}/{thousands(len(sel))} [{100 * upper95(k, len(sel)):.1f}\\%]")
            else:
                cells.append(f"{pct(k, len(sel))} ({len(sel)})")
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule\\end{tabular}\\end{table*}"]
    # Headline numbers.
    scribe = [r for r in rows if r["label"] == "scribe"]
    real = [r for r in rows if r["label"] == "real"]
    synth = [r for r in rows if r["label"] == "synthetic"]
    for key, sel in (("Scribe", scribe), ("Real", real), ("Synth", synth)):
        for c in ORDER:
            s = [r for r in sel if r["condition"] == c]
            k = sum(r.get("verdict") == "generated" for r in s)
            numbers += macro(f"Twin{key}{c.replace('-', '').title()}", pct(k, len(s)))
            numbers += macro(f"Twin{key}{c.replace('-', '').title()}N", f"{k}/{len(s)}")
    k = sum(r.get("verdict") == "generated" for r in real)
    numbers += macro("TwinRealAllK", k) + macro("TwinRealAllN", f"{len(real):,}".replace(",", "{,}"))
    numbers += macro("TwinRealAllUpper", f"{100 * upper95(k, len(real)):.2f}\\%")
    k = sum(r.get("verdict") == "generated" for r in scribe)
    numbers += macro("TwinScribeAll", pct(k, len(scribe))) + macro("TwinScribeAllN", f"{k}/{len(scribe)}")
    current = [r for r in scribe if (r.get("scale") or 4) == 4 and r["condition"] in ("none", "jpeg", "print-scan", "photo")]
    k = sum(r.get("verdict") == "generated" for r in current)
    numbers += macro("TwinScribeCurrent", pct(k, len(current))) + macro("TwinScribeCurrentN", f"{k}/{len(current)}")
    for name, sel in (("Old", [r for r in scribe if r.get("scale") == 2]), ("New", [r for r in scribe if (r.get("scale") or 4) == 4])):
        k = sum(r.get("verdict") == "generated" for r in sel)
        numbers += macro(f"TwinScribe{name}", pct(k, len(sel))) + macro(f"TwinScribe{name}N", f"{k}/{len(sel)}")
    # Rendered pages by messiness setting (which scales the jitter the test undoes).
    rendered = [r for r in scribe if r["source"] == "generated" and r.get("messiness") is not None]
    levels = sorted({r["messiness"] for r in rendered})
    if levels:
        rates = [100 * np.mean([x.get("verdict") == "generated" for x in rendered if x["messiness"] == m]) for m in levels]
        numbers += macro("TwinMessMin", f"{min(rates):.1f}\\%") + macro("TwinMessMax", f"{max(rates):.1f}\\%")
        numbers += macro("MessLow", levels[0]) + macro("MessHigh", levels[-1]) + macro("MessLevels", len(levels))
    misses = [r for r in scribe if r.get("verdict") != "generated"]
    numbers += macro("TwinScribeMissK", len(misses)) + macro("TwinScribeMissOldK", sum(r.get("scale") == 2 for r in misses))
    numbers += macro("TwinScribeMissSuspK", sum(r.get("verdict") == "suspicious" for r in misses))
    # The softer verdicts: "suspicious" (weak reuse, or a known export layout) and "too little to judge".
    for key, sel in (("Scribe", scribe), ("Real", real), ("Synth", synth)):
        for verdict, short in (("suspicious", "Susp"), ("too-little", "TooLittle")):
            k = sum(r.get("verdict") == verdict for r in sel)
            numbers += macro(f"Twin{key}{short}", pct_fine(k, len(sel))) + macro(f"Twin{key}{short}K", k)
    for source in ("cvl", "ipad", "mw-human"):
        sel = [r for r in real if r["source"] == source]
        k = sum(r.get("verdict") == "generated" for r in sel)
        key = re.sub(r"[^A-Za-z]", "", source.title())
        numbers += macro(f"TwinReal{key}K", k) + macro(f"TwinReal{key}N", thousands(len(sel)))
        numbers += macro(f"TwinReal{key}Upper", f"{100 * upper95(k, len(sel)):.2f}\\%")
        numbers += macro(f"TwinReal{key}Susp", sum(r.get("verdict") == "suspicious" for r in sel))
    return "\n".join(lines) + "\n", numbers


def cnn(rows: list[dict]) -> tuple[str, str, dict]:
    """The CNN against the twin test, protocol by protocol, on the same documents."""
    twin = {f"{r['id']}|{r['condition']}": r for r in rows if r.get("reuse")}
    out, numbers, data = [], "", {}
    names = {"random": "Random split", "clean-only": "Clean-only training", "holdout-real_cvl": "Held-out real: CVL",
             "holdout-real_ipad": "Held-out real: tablet", "holdout-real_mw-human": "Held-out real: MathWriting",
             "holdout-gen_scribe": "Held-out generator: Scribe", "holdout-gen_mw-synthetic": "Held-out generator: MathWriting"}
    from sklearn.metrics import roc_auc_score

    out = ["\\begin{table*}[t]\\centering\\caption{The trained CNN against the glyph-reuse test, on the documents each protocol "
           "tests (all conditions pooled). False alarms: real documents called generated. Detection: generated documents called "
           "generated. A held-out real source's row counts false alarms on that source only; a held-out generator's row counts "
           "detection of that generator only. AUC is over the same documents (--: one class only).}\\label{tab:protocols}",
           "\\footnotesize\\begin{tabular}{@{}lrrrrrr@{}}\\toprule",
           " & \\multicolumn{2}{c}{AUC} & \\multicolumn{2}{c}{False alarms} & \\multicolumn{2}{c}{Detection} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
           "Protocol & Twin & CNN & Twin & CNN & Twin & CNN \\\\\\midrule"]
    for stem, label in names.items():
        path = RUNS / "learned" / f"{stem}.json"
        if not path.exists():
            continue
        docs = json.loads(path.read_text())["docs"]
        docs = [d for d in docs if f"{d['doc']}|{d['condition']}" in twin]
        if stem.startswith("holdout-real_"):
            held = stem.split("_", 1)[1]
            scope = [d for d in docs if d["source"] == held or d["label"] in POSITIVE]
            neg_scope = [d for d in docs if d["source"] == held]
            pos_scope = [d for d in docs if d["label"] in POSITIVE]
        elif stem.startswith("holdout-gen_"):
            held = stem.split("_", 1)[1]
            pos_scope = [d for d in docs if d["source"] == held or (held == "scribe" and d["label"] == "scribe")]
            neg_scope = [d for d in docs if d["label"] == "real"]
            scope = pos_scope + neg_scope
        else:
            scope = docs
            neg_scope = [d for d in docs if d["label"] == "real"]
            pos_scope = [d for d in docs if d["label"] in POSITIVE]
        def t_flag(d):
            return twin[f"{d['doc']}|{d['condition']}"]["verdict"] == "generated"

        def m_flag(d):
            return d["score"] >= 0.5

        def auc(sel, score):
            y = [int(d["label"] in POSITIVE) for d in sel]
            return f"{roc_auc_score(y, [score(d) for d in sel]):.3f}" if len(set(y)) == 2 else "--"

        def t_score(d):
            p = twin[f"{d['doc']}|{d['condition']}"]["reuse"]["p"]  # 0.0 when the binomial tail underflows: the strongest case
            return -np.log10(max(1.0 if p is None else p, 1e-300))

        t_auc, m_auc = auc(scope, t_score), auc(scope, lambda d: d["score"])
        t_fa, m_fa = sum(map(t_flag, neg_scope)), sum(map(m_flag, neg_scope))
        t_det, m_det = sum(map(t_flag, pos_scope)), sum(map(m_flag, pos_scope))
        data[stem] = dict(t_auc=t_auc, m_auc=m_auc, t_fa=t_fa, m_fa=m_fa, neg=len(neg_scope), t_det=t_det, m_det=m_det, pos=len(pos_scope))
        out.append(f"{label} & {t_auc} & {m_auc} & {pct_fine(t_fa, len(neg_scope))} & {pct(m_fa, len(neg_scope))} & "
                   f"{pct(t_det, len(pos_scope))} & {pct(m_det, len(pos_scope))} \\\\")
        key = re.sub(r"[^A-Za-z]", "", stem.title())
        numbers += (macro(f"Cnn{key}Auc", m_auc) + macro(f"Twin{key}Auc", t_auc) +
                    macro(f"Cnn{key}Fa", pct(m_fa, len(neg_scope))) + macro(f"Twin{key}Fa", pct_fine(t_fa, len(neg_scope))) +
                    macro(f"Cnn{key}Det", pct(m_det, len(pos_scope))) + macro(f"Twin{key}Det", pct(t_det, len(pos_scope))) +
                    macro(f"{key}Docs", thousands(len({d['doc'] for d in scope}))) + macro(f"{key}Train", thousands(json.loads(path.read_text())["train_pages"])))
        # By condition (false alarms on the scope's real documents, detection of its generated ones) and by source.
        for c in ORDER:
            cond = c.replace("-", "").title()
            neg_c = [d for d in neg_scope if d["condition"] == c]
            pos_c = [d for d in pos_scope if d["condition"] == c]
            numbers += (macro(f"Cnn{key}Fa{cond}", pct(sum(map(m_flag, neg_c)), len(neg_c))) +
                        macro(f"Twin{key}Fa{cond}", pct_fine(sum(map(t_flag, neg_c)), len(neg_c))) +
                        macro(f"Cnn{key}Det{cond}", pct(sum(map(m_flag, pos_c)), len(pos_c))) +
                        macro(f"Twin{key}Det{cond}", pct(sum(map(t_flag, pos_c)), len(pos_c))))
        for source in sorted({d["source"] for d in docs}):
            sel = [d for d in docs if d["source"] == source]
            src = re.sub(r"[^A-Za-z]", "", source.title())
            numbers += (macro(f"Cnn{key}On{src}", pct(sum(map(m_flag, sel)), len(sel))) +
                        macro(f"Twin{key}On{src}", pct_fine(sum(map(t_flag, sel)), len(sel))))
    out.append("\\bottomrule\\end{tabular}\\end{table*}")
    return "\n".join(out) + "\n", numbers, data


def side_numbers() -> str:
    numbers = ""
    report = RUNS / "printed" / "report.md"
    if report.exists():
        m = re.search(r"Printed type recalled: ([\d.]+)% of (\d+); handwriting called printed: ([\d.]+)%", report.read_text())
        if m:
            numbers += macro("PrintRecall", f"{float(m.group(1)):.1f}\\%") + macro("PrintFalse", f"{float(m.group(3)):.1f}\\%")
    texture = RUNS / "texture" / "summary.json"
    if texture.exists():
        t = json.loads(texture.read_text())
        numbers += macro("TexAucNone", f"{t.get('cv:none', float('nan')):.3f}")
        numbers += macro("TexAucPhoto", f"{t.get('cv:photo', float('nan')):.3f}")
        numbers += macro("TexHeldCvl", f"{100 * (t.get('heldout-real:cvl') or 0):.1f}\\%")
        numbers += macro("TexHeldIpad", f"{100 * (t.get('heldout-real:ipad') or 0):.1f}\\%")
    null = RUNS / "mw-null" / "summary.json"
    if null.exists():
        s = json.loads(null.read_text())
        at = s.get("0.08", {})
        numbers += macro("MwExpressions", f"{s['expressions']:,}".replace(",", "{,}")) + macro("MwComplex", f"{s['complex']:,}".replace(",", "{,}"))
        numbers += macro("MwCopies", at.get("extra_copies", at.get("twinned", 0)))
        numbers += macro("MwRate", f"{1000 * at.get('rate', 0):.2f}") + macro("MwUpper", f"{1000 * at.get('upper95', 0):.2f}")
    dist = RUNS / "distances" / "distances.json"
    if dist.exists():
        d = json.loads(dist.read_text())
        groups = {k: d.get(k, []) for k in ("real", "scribe", "synthetic")}
        groups.update({re.sub(r"[^A-Za-z]", "", s.title()): v for s, v in d.get("by_source", {}).items()})
        for key, vals in groups.items():
            vals = np.array(vals, float)
            if not len(vals):
                continue
            name = key[:1].upper() + key[1:]
            near = vals[vals < 9]  # letters with a look-alike close enough to align
            numbers += macro(f"Dist{name}Twin", pct(int(np.sum(vals < 0.08)), len(vals)))
            numbers += macro(f"Dist{name}Letters", thousands(len(vals)))
            numbers += macro(f"Dist{name}Median", f"{np.median(near):.2f}" if len(near) else "--")
            numbers += macro(f"Dist{name}Aligned", pct(len(near), len(vals), 0))
        numbers += macro("DistDocs", thousands(sum(d.get("documents", {}).values())))
        numbers += macro("DistSetAside", thousands(sum(d.get("set_aside", {}).values())))
    cal = json.loads((ROOT / "inkcheck" / "calibration.json").read_text())
    numbers += macro("NullRatePerK", f"{1000 * cal['null_rate']:.1f}") + macro("TwinThreshold", cal["twin_displacement"])
    return numbers


# ---------------------------------------------------------------------------
# Figures


def figures(rows: list[dict], comparison: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    blue, orange, grey = "#2f5bd3", "#d97706", "#6b7280"

    # Fig: share of complex letters whose closest look-alike lies within stroke distance d (cumulative).
    dist = RUNS / "distances" / "distances.json"
    if dist.exists():
        d = json.loads(dist.read_text())
        fig, ax = plt.subplots(figsize=(3.4, 2.1))
        grid = np.linspace(0, 0.5, 501)
        for key, colour, label in (("scribe", blue, "Scribe"), ("synthetic", orange, "MathWriting synthetic"), ("real", grey, "real handwriting")):
            vals = np.array(d.get(key, []), float)
            if len(vals):
                ax.plot(grid, [100 * np.mean(vals <= x) for x in grid], color=colour, linewidth=1.2, label=f"{label} ({len(vals):,} letters)")
        ax.axvline(0.08, color="black", linewidth=0.6, linestyle="--")
        ax.text(0.088, 92, "twin threshold", fontsize=6.5)
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 100)
        ax.set_xlabel("distance $d$ to the closest look-alike (pen widths)")
        ax.set_ylabel("% of letters within $d$")
        ax.legend(frameon=False, fontsize=6.5, loc="center right")
        fig.tight_layout()
        fig.savefig(FIG / "distance.pdf")
        plt.close(fig)

    # Fig: by condition, glyph-reuse test vs CNN within the CNN's training distribution (random protocol).
    path = RUNS / "learned" / "random.json"
    if path.exists():
        docs = json.loads(path.read_text())["docs"]
        twin = {f"{r['id']}|{r['condition']}": r for r in rows}
        docs = [d for d in docs if f"{d['doc']}|{d['condition']}" in twin]
        panels = (("Scribe documents caught", lambda d: d["label"] == "scribe"),
                  ("MathWriting synthetic caught", lambda d: d["label"] == "synthetic"),
                  ("Real documents called generated", lambda d: d["label"] == "real"))
        fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
        x = np.arange(len(ORDER))
        for ax, (title, want) in zip(axes, panels):
            t_vals, m_vals = [], []
            for c in ORDER:
                sel = [d for d in docs if d["condition"] == c and want(d)]
                t_vals.append(100 * np.mean([twin[f"{d['doc']}|{c}"]["verdict"] == "generated" for d in sel]) if sel else 0)
                m_vals.append(100 * np.mean([d["score"] >= 0.5 for d in sel]) if sel else 0)
            ax.bar(x - 0.2, t_vals, 0.38, color=blue, label="glyph-reuse test")
            ax.bar(x + 0.2, m_vals, 0.38, color=orange, label="trained CNN")
            for xi, (tv, mv) in enumerate(zip(t_vals, m_vals)):
                ax.text(xi - 0.2, tv + 2, f"{tv:.0f}", ha="center", va="bottom", rotation=90, fontsize=5.5)
                ax.text(xi + 0.2, mv + 2, f"{mv:.0f}", ha="center", va="bottom", rotation=90, fontsize=5.5)
            ax.set_xticks(x, [COND_NAME[c] for c in ORDER], rotation=30, ha="right", fontsize=6.5)
            ax.set_ylim(0, 122)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title(title, fontsize=8)
        axes[0].set_ylabel("% of documents")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center", ncol=2)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(FIG / "conditions.pdf")
        plt.close(fig)

    # Fig: generalisation, false alarms on a real source held out of training, and detection of a held-out generator.
    if comparison:
        items = [("holdout-real_cvl", "CVL scans", "fa"), ("holdout-real_ipad", "tablet notes", "fa"), ("holdout-real_mw-human", "MathWriting human", "fa"),
                 ("holdout-gen_scribe", "Scribe", "det"), ("holdout-gen_mw-synthetic", "MathWriting synth.", "det")]
        items = [i for i in items if i[0] in comparison]
        if items:
            fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [3, 2]})
            for ax, kind, title in ((axes[0], "fa", "False alarms on a real source held out of training"),
                                    (axes[1], "det", "Detection of a generator held out of training")):
                sel = [i for i in items if i[2] == kind]
                x = np.arange(len(sel))
                for off, who, colour, label in ((-0.2, "t", blue, "glyph-reuse test"), (0.2, "m", orange, "trained CNN")):
                    vals = [100 * comparison[s][f"{who}_{kind}"] / max(comparison[s]["neg" if kind == "fa" else "pos"], 1) for s, _, _ in sel]
                    ax.bar(x + off, vals, 0.38, color=colour, label=label)
                    for xi, v in zip(x, vals):
                        ax.text(xi + off, v + 1.5, f"{v:.0f}", ha="center", fontsize=6)
                ax.set_xticks(x, [n for _, n, _ in sel])
                ax.set_ylim(0, 110)
                ax.set_ylabel("% of documents")
                ax.set_title(title, fontsize=8)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, frameon=False, fontsize=7, loc="upper center", ncol=2)
            fig.tight_layout(rect=(0, 0, 1, 0.88))
            fig.savefig(FIG / "generalization.pdf")
            plt.close(fig)


# ---------------------------------------------------------------------------
# docs/results.md: the same tables and numbers, for reading in the repository.


def _plain(text: str) -> str:
    """A LaTeX table cell or caption as plain markdown text."""
    text = re.sub(r"\\emph\{([^}]*)\}", r"*\1*", text)
    text = re.sub(r"\\multicolumn\{\d+\}\{\w+\}\{([^}]*)\}", r"\1", text)
    text = re.sub(r"Table~?\\ref\{tab:(\w+)\}", lambda m: {"protocols": "the CNN table", "sweep": "the sweep table"}.get(m.group(1), "the table"), text)
    for a, b in (("\\%", "%"), ("{,}", ","), ("$", ""), ("\\ ", " "), ("~", " "), ("\\geq", "≥"), ("--", "–")):
        text = text.replace(a, b)
    return re.sub(r"\s+", " ", text).strip()


def _caption(tex: str) -> str:
    start = tex.index("\\caption{") + len("\\caption{")
    depth, i = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(tex[i], 0)
        i += 1
    return _plain(tex[start:i - 1])


def table_markdown(tex: str) -> str:
    """One of this script's generated LaTeX tables, as a markdown table under its caption."""
    body = tex[tex.index("\\toprule") + len("\\toprule"):tex.index("\\bottomrule")]
    chunks = body.split("\\\\")
    first_data = next(i for i, c in enumerate(chunks) if "\\midrule" in c)  # \midrule follows the last header row

    def cells_of(chunk: str) -> list[str]:
        chunk = re.sub(r"\\midrule|\\cmidrule\([^)]*\)\{[\d-]+\}", "", chunk).strip()
        out: list[str] = []
        for cell in chunk.split("&") if chunk else []:
            m = re.match(r"\s*\\multicolumn\{(\d+)\}", cell)
            out += [_plain(cell)] * (int(m.group(1)) if m else 1)
        return out

    header = [c for c in map(cells_of, chunks[:first_data]) if c]
    rows = [c for c in map(cells_of, chunks[first_data:]) if c]
    width = max(len(r) for r in header + rows)
    merged = [" ".join(h[c] for h in header if c < len(h) and h[c]) for c in range(width)]
    lines = [_caption(tex), "", "| " + " | ".join(merged) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def results_markdown(numbers: str, tables: list[str], run: str) -> str:
    n = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", numbers))
    v = {k: _plain(x) for k, x in n.items()}
    g = lambda key: v.get(key, "–")
    lines = [
        "# Results",
        "",
        f"Written by `docs/paper/make_figures.py` from `runs/{run}` (the glyph-reuse test) and `runs/learned` (the trained CNN). "
        "Aggregate numbers only. The paper, [paper/inkcheck.pdf](paper/inkcheck.pdf), explains every number here.",
        "",
        "## Headline",
        "",
        f"- **The glyph-reuse test** called {g('TwinScribeAll')} of Scribe's document-conditions generated "
        f"({g('TwinScribeNew')} at today's 4× export resolution, {g('TwinScribeOld')} of the older 2× exports), "
        f"across five conditions including print-and-scan and phone photos.",
        f"- **False alarms on real handwriting:** {g('TwinRealAllK')} of {g('TwinRealAllN')} document-conditions "
        f"(95% upper bound {g('TwinRealAllUpper')}); {g('TwinRealSuspK')} more were marked suspicious.",
        f"- **MathWriting's generator** (a pool of 6,423 glyphs) is the hard case: {g('TwinSynthNone')} caught as exported.",
        f"- **The trained CNN** wins inside its training distribution (AUC {g('CnnRandomAuc')} against the test's "
        f"{g('TwinRandomAuc')}) but called {g('CnnHoldoutRealCvlFa')} of CVL's pages generated when CVL was held out of "
        f"training, {g('CnnHoldoutRealIpadFa')} of the tablet pages, and {g('CnnHoldoutRealMwHumanFa')} of MathWriting's "
        f"human pages; it caught {g('CnnHoldoutGenScribeDet')} of Scribe's documents when Scribe was held out.",
        f"- **Capacity sweep:** at 0.15 M, 0.58 M and 2.33 M weights the CNN still called {g('SweepCvlHalf')}, "
        f"{g('SweepCvlOne')} and {g('SweepCvlTwo')} of held-out CVL generated.",
        f"- **Same writer:** {g('MwExpressions')} single MathWriting expressions held {g('MwComplex')} complex letters and "
        f"{g('MwCopies')} extra copies (95% upper bound {g('MwUpper')} per 1,000).",
        f"- **Closest look-alikes:** {g('DistScribeTwin')} of Scribe's complex letters have a twin under 0.08 pen widths; "
        f"{g('DistRealTwin')} of real letters do.",
        "- **What the real copies are:** of the 49 groups of chance copies on CVL (all 43 pages that had any), 48 were printed "
        "type the print rules missed (43 scan-broken bowls of a printed g, five printed 7s) and one was handwriting (two "
        "cursive l's at d = 0.072).",
        "",
        "## Tables",
        "",
    ]
    headline = lines[lines.index("## Headline") + 2:lines.index("## Tables") - 1]
    readme = ROOT / "README.md"
    text = readme.read_text()
    if "<!-- RESULTS -->" in text:
        block = ("<!-- RESULTS -->\n## Results\n\n" + "\n".join(headline) + "\n\nAll numbers: [docs/results.md](docs/results.md). "
                 "The paper: [docs/paper/inkcheck.pdf](docs/paper/inkcheck.pdf).\n<!-- /RESULTS -->")
        end = text.find("<!-- /RESULTS -->")
        text = text[:text.index("<!-- RESULTS -->")] + block + (text[end + len("<!-- /RESULTS -->"):] if end >= 0 else text[text.index("<!-- RESULTS -->") + len("<!-- RESULTS -->"):])
        readme.write_text(text)
    for tex in tables:
        if "\\toprule" in tex:
            lines += [table_markdown(tex), ""]
    return "\n".join(lines)


def main() -> None:
    rows = load_twin(sys.argv[1] if len(sys.argv) > 1 else None)
    if not rows:
        sys.exit("no benchmark results in runs/ yet")
    data_table, n1 = dataset(rows)
    twin_tab, n2 = twin_table(rows)
    cnn_tab, n3, comparison = cnn(rows)
    sens_tab, n4 = sensitivity(rows)
    # The recomputed verdict must agree with the detector's own at the pre-set rate.
    cal = json.loads((ROOT / "inkcheck" / "calibration.json").read_text())
    disagree = [r["key"] for r in rows if r.get("reuse") and flagged(r, cal["null_rate"], cal["min_copies"]) != (r.get("verdict") == "generated")]
    if disagree:
        print(f"warning: recomputed verdict differs from the detector's on {len(disagree)} rows, e.g. {disagree[:3]}")
    sweep_tab, n5 = sweep()
    numbers = ("% Written by make_figures.py from runs/. Do not edit by hand.\n" + n1 + n2 + n3 + n4 + n5 + prefix_numbers()
               + side_numbers())
    (HERE / "numbers.tex").write_text(numbers)
    (HERE / "table_data.tex").write_text(data_table)
    (HERE / "table_twin.tex").write_text(twin_tab)
    (HERE / "table_protocols.tex").write_text(cnn_tab)
    (HERE / "table_sensitivity.tex").write_text(sens_tab)
    (HERE / "table_sweep.tex").write_text(sweep_tab or "% sweep not run yet\n")
    figures(rows, comparison)
    run = sys.argv[1] if len(sys.argv) > 1 else ("final" if (RUNS / "final" / "results.jsonl").exists() else "main")
    (ROOT / "docs" / "results.md").write_text(results_markdown(numbers, [twin_tab, sens_tab, cnn_tab, sweep_tab, data_table], run))
    print(f"numbers.tex: {numbers.count(chr(10))} lines; tables and figures written to {HERE}; docs/results.md")


if __name__ == "__main__":
    main()
