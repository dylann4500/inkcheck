"""inkcheck's command line.

    inkcheck scan FILE... [--html PATH] [--json PATH] [--layers provenance,reuse,texture] [--workers N] [--quiet]
    inkcheck serve [--port 8765]

Exit status of `scan`: 2 if any file is generated, else 1 if any is
suspicious, else 3 if any could not be judged (too little, or unreadable),
else 0.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .scan import ALL_LAYERS, scan

EXIT = {"generated": 2, "suspicious": 1, "too-little": 3, "error": 3, "no-evidence": 0}
MARK = {"generated": "GENERATED", "suspicious": "SUSPICIOUS", "too-little": "TOO LITTLE TO JUDGE",
        "error": "UNREADABLE", "no-evidence": "NO EVIDENCE FOUND"}


def _report_path(option: str | None, source: Path, many: bool, suffix: str, taken: set[Path] | None = None) -> Path | None:
    if not option:
        return None
    target = Path(option)
    if many or target.is_dir() or not target.suffix:
        target.mkdir(parents=True, exist_ok=True)
        path = target / (source.stem + suffix)
        if taken is not None:
            # Two inputs with one name (a/hw.png, b/hw.png) must not share a report.
            if path in taken:
                path = target / f"{source.parent.name or 'file'}-{source.stem}{suffix}"
            number = 2
            while path in taken:
                path = target / f"{source.stem}-{number}{suffix}"
                number += 1
            taken.add(path)
        return path
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def cmd_scan(args) -> int:
    from .report import write_html, write_json

    layers = tuple(x.strip() for x in args.layers.split(",") if x.strip())
    unknown = set(layers) - set(ALL_LAYERS)
    if unknown:
        print(f"Unknown layer(s): {', '.join(sorted(unknown))}. Choose from {', '.join(ALL_LAYERS)}.", file=sys.stderr)
        return 3
    files = [Path(f) for f in args.files]
    many = len(files) > 1
    labels = []
    taken_html: set[Path] = set()
    taken_json: set[Path] = set()
    for path in files:
        started = time.time()
        result = scan(path, layers, args.workers)
        labels.append(result.verdict.label)
        html_path = _report_path(args.html, path, many, ".html", taken_html)
        json_path = _report_path(args.json, path, many, ".json", taken_json)
        if html_path:
            write_html(result, html_path)
        if json_path:
            write_json(result, json_path)
        if not args.quiet:
            print(f"{path.name} — {MARK[result.verdict.label]}  ({len(result.document.pages)} pages, {time.time() - started:.1f}s)")
            print(f"  {result.verdict.headline}")
            for reason in result.verdict.reasons[1:]:
                print(f"  · {reason}")
            if html_path:
                print(f"  report: {html_path}")
    if "generated" in labels:
        return 2
    if "suspicious" in labels:
        return 1
    if any(label in ("too-little", "error") for label in labels):
        return 3
    return 0


def cmd_serve(args) -> int:
    from .serve import serve

    serve(args.port, args.workers)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="inkcheck", description="Finds handwriting that software assembled from samples.")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("scan", help="check PDFs or page images")
    s.add_argument("files", nargs="+")
    s.add_argument("--html", help="write an HTML evidence report (a file, or a folder for several files)")
    s.add_argument("--json", help="write the results as JSON (a file, or a folder for several files)")
    s.add_argument("--layers", default=",".join(ALL_LAYERS), help="which checks to run (default: all)")
    s.add_argument("--workers", type=int, default=None, help="processes for comparing letters (default: up to 8)")
    s.add_argument("-q", "--quiet", action="store_true")
    s.set_defaults(run=cmd_scan)
    v = sub.add_parser("serve", help="a local web page to check files")
    v.add_argument("--port", type=int, default=8765)
    v.add_argument("--workers", type=int, default=None)
    v.set_defaults(run=cmd_serve)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main())
