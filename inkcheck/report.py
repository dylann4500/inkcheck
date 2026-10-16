"""The report: JSON for programs, and one self-contained HTML page for people.

The HTML page shows the evidence itself, not just scores: every group of
letters that are copies of one another, cut from the page side by side, and
the closest pair laid over each other (one in red, one in blue, black where
they agree), with the turn, stretch and slant that separate them. It embeds
the pages' handwriting, so it is written only where the person asks.
"""

from __future__ import annotations

import base64
import html
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .scan import ScanResult

CLUSTER_COLOURS = [(230, 57, 70), (29, 120, 219), (46, 160, 67), (217, 119, 6), (147, 51, 234), (13, 148, 136),
                   (219, 39, 119), (101, 163, 13), (2, 132, 199), (180, 83, 9), (124, 58, 237), (225, 29, 72)]
VERDICT_TEXT = {
    "generated": "Generated",
    "suspicious": "Suspicious",
    "no-evidence": "No evidence found",
    "too-little": "Too little to judge",
    "error": "Could not read",
}


# ---------------------------------------------------------------------------
# JSON


def to_dict(result: ScanResult) -> dict:
    doc = result.document
    return {
        "file": str(doc.path),
        "inkcheck": __version__,
        "pages": len(doc.pages),
        "verdict": {"label": result.verdict.label, "headline": result.verdict.headline, "reasons": result.verdict.reasons},
        "layers": {name: {"score": layer.score, "findings": [f.__dict__ for f in layer.findings], "data": layer.data,
                          "evidence": layer.evidence} for name, layer in result.layers.items()},
        "metadata": {k: v for k, v in doc.metadata.items() if k != "xmp"},
    }


def _plain(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON: {type(value)}")


def write_json(result: ScanResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_dict(result), indent=2, default=_plain))
    return path


# ---------------------------------------------------------------------------
# Images


def _data_url(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _crop(pixels: np.ndarray, box, pad: int) -> np.ndarray:
    x, y, w, h = box
    H, W = pixels.shape[:2]
    return pixels[max(0, y - pad) : min(H, y + h + pad), max(0, x - pad) : min(W, x + w + pad)]


def _to_height(rgb: np.ndarray, height: int) -> np.ndarray:
    if rgb.size == 0:
        return np.full((height, height, 3), 255, np.uint8)
    scale = height / rgb.shape[0]
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, (max(1, int(round(rgb.shape[1] * scale))), height), interpolation=method)


def _own_ink(piece, shape) -> np.ndarray:
    """A piece's own darkness on a blank page-sized canvas region (no neighbours)."""
    canvas = np.zeros(shape, np.float32)
    x0, y0 = piece.comp.origin
    h, w = piece.comp.darkness.shape
    canvas[y0 : y0 + h, x0 : x0 + w] = np.where(piece.comp.mask | (piece.comp.darkness > 0), piece.comp.darkness, 0)
    return canvas


def pair_overlay(result: ScanResult, pair: dict, height: int = 120) -> np.ndarray:
    """Piece a mapped onto piece b's place: a in red, b in blue, black where both."""
    pieces = result.reuse.pieces
    a, b = pieces[pair["a"]], pieces[pair["b"]]
    pad = int(np.ceil(0.35 * b.letter_height))
    x, y, w, h = b.comp.box
    shape_b = result.inks[b.comp.page].darkness.shape
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(shape_b[1], x + w + pad), min(shape_b[0], y + h + pad)
    region_b = _own_ink(b, shape_b)[y0:y1, x0:x1]
    m = np.array(pair["matrix"], np.float64).copy()
    m[0, 2] -= x0
    m[1, 2] -= y0
    own_a = _own_ink(a, result.inks[a.comp.page].darkness.shape)
    region_a = cv2.warpAffine(own_a, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)

    def unit(img):
        ink = img[img > 0.05]
        return np.clip(img / max(float(np.percentile(ink, 90)) if ink.size else 1.0, 1e-3), 0, 1)

    da, db = unit(region_a), unit(region_b)
    rgb = np.empty(db.shape + (3,), np.float32)
    rgb[..., 0] = 1 - db          # red channel falls where b (blue) is
    rgb[..., 1] = 1 - np.maximum(da, db)
    rgb[..., 2] = 1 - da          # blue channel falls where a (red) is
    return _to_height((rgb * 255).clip(0, 255).astype(np.uint8), height)


def page_thumbnail(result: ScanResult, page: int, width: int = 760) -> np.ndarray:
    pixels = result.document.pages[page].pixels
    scale = width / pixels.shape[1]
    thumb = cv2.resize(pixels, (width, int(round(pixels.shape[0] * scale))), interpolation=cv2.INTER_AREA).copy()
    reuse = result.layers.get("reuse")
    if not reuse:
        return thumb
    colour_index = 0
    for item in reuse.evidence:
        if item["kind"] == "reused":
            colour = CLUSTER_COLOURS[colour_index % len(CLUSTER_COLOURS)]
            colour_index += 1
        else:
            colour = (234, 88, 12) if item["kind"] == "block" else (120, 120, 120)
        for member in item["members"]:
            if member["page"] != page:
                continue
            x, y, w, h = member["box"]
            cv2.rectangle(thumb, (int(x * scale) - 2, int(y * scale) - 2), (int((x + w) * scale) + 2, int((y + h) * scale) + 2),
                          colour, 2)
    return thumb


# ---------------------------------------------------------------------------
# HTML

CSS = """
:root{--bg:#fbfaf7;--card:#ffffff;--ink:#1d1f24;--muted:#5d6370;--line:#e6e2da;--gen:#c2272d;--sus:#b45309;
--ok:#2f6f4f;--none:#6b7280;--chip:#f2efe8}
@media (prefers-color-scheme:dark){:root{--bg:#15161a;--card:#1d1f24;--ink:#ecebe7;--muted:#a2a6af;--line:#30333a;
--gen:#f2545b;--sus:#f59e0b;--ok:#5fbf8a;--none:#9ca3af;--chip:#26282e}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,
"Segoe UI",Inter,Roboto,sans-serif}main{max-width:1060px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:34px 0 10px}h3{font-size:15px;margin:0 0 6px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:12px 0}
.verdict{border-left:6px solid var(--none)}.verdict.generated{border-left-color:var(--gen)}.verdict.suspicious{border-left-color:var(--sus)}
.verdict.no-evidence{border-left-color:var(--ok)}.badge{display:inline-block;font-weight:650;font-size:13px;letter-spacing:.02em;
text-transform:uppercase;padding:2px 10px;border-radius:999px;color:#fff;background:var(--none)}.badge.generated{background:var(--gen)}
.badge.suspicious{background:var(--sus)}.badge.no-evidence{background:var(--ok)}.headline{font-size:18px;font-weight:600;margin:8px 0}
ul{margin:6px 0 0 18px;padding:0}li{margin:3px 0}.chip{display:inline-block;background:var(--chip);border-radius:6px;padding:0 7px;
font-size:12px;margin-right:6px;color:var(--muted)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px}
.cluster{display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap}.crops{display:flex;gap:6px;flex-wrap:wrap;align-items:flex-end}
.crops img{height:64px;border:1px solid var(--line);border-radius:4px;background:#fff}.overlay img{height:120px;border:1px solid var(--line);
border-radius:6px;background:#fff}.swatch{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.page img{width:100%;border:1px solid var(--line);border-radius:6px;background:#fff}table{border-collapse:collapse;width:100%}
td{border-top:1px solid var(--line);padding:5px 8px;vertical-align:top}td:first-child{color:var(--muted);width:38%}
.small{font-size:13px}
"""


def _esc(text) -> str:
    return html.escape(str(text))


def render_html(result: ScanResult) -> str:
    doc, verdict = result.document, result.verdict
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>inkcheck · {_esc(doc.path.name)}</title><style>{CSS}</style></head><body><main>",
             f"<div class='muted small'>inkcheck {__version__} · {datetime.now():%Y-%m-%d %H:%M}</div>",
             f"<h1>{_esc(doc.path.name)}</h1><div class='muted'>{len(doc.pages)} page{'s' if len(doc.pages) != 1 else ''}</div>",
             f"<div class='card verdict {verdict.label}'><span class='badge {verdict.label}'>{VERDICT_TEXT[verdict.label]}</span>"
             f"<div class='headline'>{_esc(verdict.headline)}</div>"]
    if verdict.reasons:
        parts.append("<ul>" + "".join(f"<li>{_esc(r)}</li>" for r in verdict.reasons) + "</ul>")
    parts.append("<p class='muted small'>inkcheck reports evidence, never a person's guilt: it says when handwriting was assembled "
                 "from samples, and it never says that a page was written by hand.</p></div>")

    reuse = result.layers.get("reuse")
    if reuse is not None and result.reuse is not None:
        reused = [e for e in reuse.evidence if e["kind"] == "reused"]
        if reused:
            parts.append("<h2>Letters that are copies of one another</h2><p class='muted'>Each row is one shape found more than once. "
                         "On the right, the closest two are laid over each other once a small turn, slant or stretch is undone: "
                         "red is one, blue the other, black where they agree. Written twice by a person, a letter's strokes typically "
                         "sit 0.15–0.4 pen widths apart; these sit far closer.</p>")
            for index, item in enumerate(reused[:14]):
                parts.append(_cluster_html(result, item, CLUSTER_COLOURS[index % len(CLUSTER_COLOURS)]))
        for kind, title in (("block", "Blocks of ink that appear twice"), ("printed", "Printed text (left out of the test)")):
            items = [e for e in reuse.evidence if e["kind"] == kind]
            if items:
                parts.append(f"<h2>{title}</h2>")
                for item in items[:3]:
                    parts.append(_cluster_html(result, item, (120, 120, 120), overlay=kind == "block"))

    parts.append("<h2>What each check found</h2><div class='grid'>")
    names = {"reuse": "Repeated letters", "provenance": "The file itself", "texture": "Ink texture"}
    for name, layer in result.layers.items():
        score = "not judged" if layer.score is None else f"{layer.score:.2f}"
        items = "".join(f"<li><span class='chip'>{_esc(f.weight)}</span>{_esc(f.message)}</li>" for f in layer.findings)
        parts.append(f"<div class='card'><h3>{names.get(name, name)}</h3><div class='muted small'>score {score}</div><ul>{items}</ul></div>")
    parts.append("</div>")

    provenance = result.layers.get("provenance")
    if provenance is not None:
        rows = [("Made by", provenance.data.get("producer") or "not stated"), ("Looks like", provenance.data.get("source")),
                ("Known generator layout", provenance.data.get("signature") or "none"),
                ("Pages stored as one image", provenance.data.get("raster_pages")), ("Pages with vector ink", provenance.data.get("vector_pages")),
                ("Image codecs", ", ".join(provenance.data.get("codecs") or []) or "none"),
                ("Hidden recognised text (characters)", provenance.data.get("recognised_text_chars")),
                ("Printed text (characters)", provenance.data.get("printed_text_chars"))]
        if reuse is not None:
            rows += [("Complex letters compared", reuse.data.get("complex")), ("Letters with a twin", reuse.data.get("duplicated")),
                     ("Twin threshold (pen widths)", reuse.data.get("threshold")), ("Pen width (px)", round(reuse.data.get("pen_width_px", 0), 2)),
                     ("Chance on real handwriting", f"{reuse.data.get('p_value', 1):.1e}")]
        parts.append("<h2>Facts</h2><div class='card'><table>" + "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in rows)
                     + "</table></div>")

    if result.inks and doc.pages:
        parts.append("<h2>Pages</h2><p class='muted'>Boxes mark the letters that have twins, one colour per shape.</p>")
        for page in range(min(len(doc.pages), 12)):
            parts.append(f"<div class='card page'><div class='muted small'>Page {page + 1}</div><img alt='page {page + 1}' "
                         f"src='{_data_url(page_thumbnail(result, page))}'></div>")

    parts.append("<h2>How this works</h2><div class='card small'><p>inkcheck cuts the handwriting into pieces, keeps the "
                 "pieces long and curved enough that no two written by hand come out the same (a dot, a dash or a straight "
                 "stroke can), and lines each up with its closest look-alikes. Software that assembles handwriting from "
                 "a few samples per letter leaves twins that line up almost perfectly; a person's letters never do. "
                 "The count of twins is compared with the rate measured on real handwriting. The file's own facts (which "
                 "program made it, how its pages are stored) are reported too, but they can only make a page worth a closer "
                 "look: they are easy to change, and printing and scanning erase them.</p></div></main></body></html>")
    return "".join(parts)


def _cluster_html(result: ScanResult, item: dict, colour, overlay: bool = True) -> str:
    pieces = result.reuse.pieces
    crops = []
    for member in item["members"][:8]:
        piece = pieces[member["piece"]]
        pad = int(np.ceil(0.25 * piece.letter_height))
        crop = _crop(result.document.pages[member["page"]].pixels, member["box"], pad)
        crops.append(f"<img alt='copy on page {member['page'] + 1}' title='page {member['page'] + 1}' src='{_data_url(_to_height(crop, 96))}'>")
    more = len(item["members"]) - 8
    caption = ""
    overlay_html = ""
    if overlay and item["pairs"]:
        best = item["pairs"][0]
        overlay_html = f"<div class='overlay'><img alt='closest pair laid over each other' src='{_data_url(pair_overlay(result, best))}'></div>"
        caption = (f"Closest pair: strokes {best['displacement']:.3f} pen widths apart after a turn of {best['turn_deg']:+.1f}°, "
                   f"a scale of {best['scale']:.3f}× and a slant of {best['shear']:+.3f}.")
    swatch = f"<span class='swatch' style='background:rgb{tuple(colour)}'></span>"
    pages = sorted({m["page"] + 1 for m in item["members"]})
    return (f"<div class='card'><div class='small muted'>{swatch}{len(item['members'])} copies · page{'s' if len(pages) > 1 else ''} "
            f"{', '.join(map(str, pages))}</div><div class='cluster'><div class='crops'>{''.join(crops)}"
            f"{f'<span class=muted>+{more} more</span>' if more > 0 else ''}</div>{overlay_html}</div>"
            f"<div class='small muted'>{_esc(caption)}</div></div>")


def write_html(result: ScanResult, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render_html(result), encoding="utf-8")
    return path
