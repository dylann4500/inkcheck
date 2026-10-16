"""Builds data/manifest.jsonl: every file the evaluation uses, with its truth.

Rows: {id, path, label: "real" | "scribe" | "synthetic", source, writer, hand, messiness,
paper, ruling, scale, worksheet}. Sources:

- `generated`: Scribe pages rendered with Scribe's own renderer (data/scribe/renders.jsonl).
  The generator drives Scribe's private code and is not part of this repository.
- `export`: Scribe's own Export PDFs found in the local folder (INKCHECK_LOCAL, else
  ~/Downloads), recognised by Scribe's export layout.
- `ipad`: real tablet-note exports, listed in data/real_files.json as
  {"file name.pdf": "writer id"} and read from the same folder. The list, like the
  handwriting itself, stays on this machine.
- `cvl`: the CVL database's scanned forms (310 writers), when it has been
  downloaded and unpacked under data/public/cvl/.
- `mw-human`, `mw-synthetic`: worksheet-like pages from Google's MathWriting
  (inkcheck.bench.mathwriting): people's handwritten math (real), and
  expressions stitched from their glyphs (another generator, label "synthetic").
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOCAL = Path(os.environ.get("INKCHECK_LOCAL", Path.home() / "Downloads"))


def real_files() -> dict[str, str]:
    """Real tablet exports in the local folder and who wrote them (data/real_files.json, never committed)."""
    listing = ROOT / "data" / "real_files.json"
    return json.loads(listing.read_text()) if listing.exists() else {}


EXPORT_PATTERN = re.compile(r"\.pdf$", re.I)


def _digest(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:16]


def generated_rows() -> list[dict]:
    renders = ROOT / "data" / "scribe" / "renders.jsonl"
    rows = []
    if not renders.exists():
        return rows
    for line in renders.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        path = ROOT / r["path"]
        if not path.exists():
            continue
        rows.append({"id": f"gen:{path.stem}", "path": str(path), "label": "scribe", "source": "generated",
                     "writer": r.get("writer"), "hand": r.get("hand"), "hand_kind": r.get("kind"),
                     "messiness": r.get("messiness"), "paper": r.get("paper"), "ruling": r.get("ruling"),
                     "scale": r.get("scale"), "worksheet": r.get("worksheet"), "pages": r.get("pages")})
    return rows


def export_rows() -> list[dict]:
    from ..load import load_document
    from ..provenance import scribe_signature

    import pymupdf

    rows, seen = [], set()
    for path in sorted(LOCAL.glob("*.pdf")):
        try:  # only the producer is read before deciding to open the file properly
            with pymupdf.open(path) as pdf:
                producer = (pdf.metadata or {}).get("producer") or ""
        except Exception:
            continue
        if "pdf-lib" not in producer:
            continue
        doc = load_document(path)
        if doc.error or not scribe_signature(doc):
            continue
        digest = _digest(path)
        if digest in seen:
            continue
        seen.add(digest)
        scale = 4 if doc.pages[0].pixels.shape[1] == 2448 else 2
        rows.append({"id": f"export:{path.stem}", "path": str(path), "label": "scribe", "source": "export", "writer": "author",
                     "hand": "own", "scale": scale, "pages": len(doc.pages)})
    return rows


def real_rows() -> list[dict]:
    rows, seen = [], set()
    for name, writer in real_files().items():  # duplicated downloads (same bytes) are dropped by hash
        path = LOCAL / name
        if not path.exists():
            continue
        digest = _digest(path)
        if digest in seen:
            continue
        seen.add(digest)
        rows.append({"id": f"ipad:{path.stem}", "path": str(path), "label": "real", "source": "ipad", "writer": writer})
    return rows


def cvl_rows() -> list[dict]:
    base = ROOT / "data" / "public" / "cvl"
    rows = []
    for path in sorted(base.rglob("*")):
        if path.suffix.lower() not in (".tif", ".tiff", ".png", ".jpg") or "cropped" in str(path).lower():
            continue
        m = re.match(r"(\d{4})-(\d+)", path.stem)
        if not m:
            continue
        rows.append({"id": f"cvl:{path.stem}", "path": str(path), "label": "real", "source": "cvl", "writer": f"cvl-{m.group(1)}",
                     "text": int(m.group(2))})
    return rows


def mathwriting_rows() -> list[dict]:
    """Pages built from MathWriting by inkcheck.bench.mathwriting (human: real; synthetic: another generator)."""
    pages = ROOT / "data" / "public" / "mathwriting-pages" / "pages.jsonl"
    if not pages.exists():
        return []
    return [json.loads(line) for line in pages.read_text().splitlines() if line.strip()]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(ROOT / "data" / "manifest.jsonl"))
    args = parser.parse_args(argv)
    rows = generated_rows() + export_rows() + real_rows() + cvl_rows() + mathwriting_rows()
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    print(f"{len(rows)} files -> {args.out}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
