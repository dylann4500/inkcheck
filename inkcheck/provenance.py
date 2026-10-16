"""Layer 1: what the file says about how it was made.

Three kinds of fact: the PDF's or image's metadata (which program wrote it),
how each page is stored (one image, vector ink, a text layer), and what the
pixels say about where they came from (a camera or scanner leaves noise and
uneven light; software leaves exactly uniform paper and a small palette).

This layer is context, not proof. Metadata is easy to rewrite, a screenshot
is a rendered raster too, and printing and scanning erases all of it. A
known generator's exact export format is the strongest thing it can say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np

from .layers import LayerResult
from .load import Document, Page

# (pattern, category, description). The first match wins.
PRODUCERS: list[tuple[str, str, str]] = [
    (r"GoodNotes", "note-app", "GoodNotes"),
    (r"Notability", "note-app", "Notability"),
    (r"OneNote", "note-app", "Microsoft OneNote"),
    (r"Samsung Notes", "note-app", "Samsung Notes"),
    (r"Xournal", "note-app", "Xournal++"),
    (r"Squid", "note-app", "Squid"),
    (r"^iOS .*Quartz PDFContext|^iPadOS .*Quartz", "apple-device", "an iPad or iPhone app (Notability, GoodNotes, Notes, Files and others export this way)"),
    (r"Mac OS X .*Quartz PDFContext|macOS .*Quartz PDFContext", "apple-mac", "a Mac app or Print to PDF on a Mac"),
    (r"Adobe Scan", "scanner-app", "Adobe Scan"),
    (r"CamScanner", "scanner-app", "CamScanner"),
    (r"Microsoft Lens|Office Lens", "scanner-app", "Microsoft Lens"),
    (r"Genius Scan", "scanner-app", "Genius Scan"),
    (r"Scanbot", "scanner-app", "Scanbot"),
    (r"TurboScan", "scanner-app", "TurboScan"),
    (r"Google", "scanner-app", "a Google app (Drive scan and others)"),
    (r"Canon|RICOH|Ricoh|Xerox|KONICA|Konica|Kyocera|KYOCERA|Brother|EPSON|Epson|HP Scan|Hewlett|SHARP|Sharp|Lexmark|PaperPort|ScanSnap|FUJITSU|Fujitsu|TOSHIBA|Toshiba|Panasonic", "scanner-device", "a scanner or copier"),
    (r"Skia/PDF", "print-to-pdf", "Chrome's Print to PDF"),
    (r"Microsoft: Print To PDF|Microsoft.*Word", "print-to-pdf", "Microsoft Print to PDF or Word"),
    (r"pdf-lib", "library", "pdf-lib, a JavaScript library web apps use to build PDFs"),
    (r"PyMuPDF|MuPDF", "library", "PyMuPDF/MuPDF"),
    (r"ReportLab", "library", "ReportLab"),
    (r"iText", "library", "iText"),
    (r"jsPDF", "library", "jsPDF"),
    (r"FPDF|TCPDF", "library", "FPDF/TCPDF"),
    (r"PDFKit", "library", "PDFKit"),
    (r"PDFsharp", "library", "PDFsharp"),
    (r"img2pdf", "library", "img2pdf"),
    (r"ImageMagick", "library", "ImageMagick"),
    (r"Ghostscript|GPL Ghostscript", "converter", "Ghostscript"),
]


@dataclass
class PixelStats:
    noise: float          # std of the paper's fine texture, grey levels
    paper_mode_share: float  # share of bright pixels that are exactly the commonest bright value
    colours: int          # distinct colours in a 1-in-16 sample
    light_falloff: float  # paper brightness range across the page, grey levels


def classify_producer(text: str) -> tuple[str, str] | None:
    for pattern, category, description in PRODUCERS:
        if text and re.search(pattern, text):
            return category, description
    return None


def pixel_stats(page: Page) -> PixelStats:
    rgb = page.pixels
    step = max(1, int(round(max(rgb.shape[:2]) / 1600)))
    small = rgb[::step, ::step]
    grey = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    bright = grey >= np.percentile(grey, 60)
    values = grey[bright]
    if values.size == 0:
        return PixelStats(0.0, 0.0, 0, 0.0)
    mode = np.bincount(values, minlength=256).argmax()
    mode_share = float(np.mean(values == mode))
    residual = grey.astype(np.int16) - cv2.medianBlur(grey, 3).astype(np.int16)
    # Paper only: bright, and away from ink edges.
    calm = bright & (cv2.morphologyEx(grey, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) <= 12)
    noise = float(np.std(residual[calm])) if calm.any() else 0.0
    sample = small.reshape(-1, 3)[::16]
    colours = int(len(np.unique(sample.astype(np.uint32) @ np.array([65536, 256, 1], np.uint32))))
    # Paper brightness in a coarse grid: a camera's uneven light shows here.
    h, w = grey.shape
    cells = []
    for gy in range(4):
        for gx in range(4):
            cell = grey[gy * h // 4 : (gy + 1) * h // 4, gx * w // 4 : (gx + 1) * w // 4]
            if cell.size:
                cells.append(np.percentile(cell, 90))
    falloff = float(max(cells) - min(cells)) if cells else 0.0
    return PixelStats(noise, mode_share, colours, falloff)


def raster_origin(stats: PixelStats) -> str:
    """`rendered` (software drew it), `captured` (a camera or scanner), or `unclear`."""
    if stats.noise < 0.6 and stats.paper_mode_share > 0.35:
        return "rendered"
    if stats.noise > 1.2 or stats.light_falloff > 18 or stats.paper_mode_share < 0.15:
        return "captured"
    return "unclear"


def scribe_signature(doc: Document) -> bool:
    """Scribe's Export PDF (`pdfFrom` in its app.js): pdf-lib, one lossless
    full-page image per page, a 612×792 page, the image at 2× or 4× of it."""
    producer = doc.metadata.get("producer", "") + " " + doc.metadata.get("creator", "")
    if doc.kind != "pdf" or "pdf-lib" not in producer or not doc.pages:
        return False
    for page in doc.pages:
        f = page.facts
        if not (f.full_page_image and f.images == 1 and f.image_codecs == ["flate"]):
            return False
        if not f.size_pt or abs(f.size_pt[0] - 612) > 0.5 or abs(f.size_pt[1] - 792) > 0.5:
            return False
        if f.image_px not in ((1224, 1584), (2448, 3168)):
            return False
    return True


SIGNATURES = {"scribe": ("Scribe's Export PDF", scribe_signature)}


def assess_provenance(doc: Document) -> LayerResult:
    result = LayerResult("provenance", None)
    if doc.error or not doc.pages:
        result.add("unreadable", doc.error or "No pages.", "info")
        return result

    who = doc.metadata.get("producer", "") or doc.metadata.get("creator", "") or doc.metadata.get("software", "")
    producer = classify_producer(" ".join(doc.metadata.get(k, "") for k in ("producer", "creator", "software")))
    camera = doc.metadata.get("camera_model") or doc.metadata.get("camera_make")
    result.data.update(producer=who or None, producer_category=producer[0] if producer else None, camera=camera)

    vector_pages = sum(1 for p in doc.pages if p.facts.vector_segments >= 500 and not p.facts.full_page_image)
    raster_pages = [p for p in doc.pages if p.facts.full_page_image]
    recognised = sum(p.facts.invisible_text_chars for p in doc.pages)
    printed = sum(p.facts.visible_text_chars for p in doc.pages)
    codecs = sorted({c for p in raster_pages for c in p.facts.image_codecs})
    origins = []
    for page in raster_pages[:4]:
        stats = pixel_stats(page)
        origins.append(raster_origin(stats))
        result.data.setdefault("pixels", []).append(stats.__dict__)

    if vector_pages:
        source = "vector-ink"
    elif raster_pages and origins.count("rendered") > len(origins) / 2:
        source = "rendered-raster"
    elif raster_pages and (origins.count("captured") > len(origins) / 2 or "jpeg" in codecs or "jbig2" in codecs or "ccitt" in codecs):
        source = "camera-or-scan"
    else:
        source = "unknown"
    signature = next((name for name, (_, test) in SIGNATURES.items() if test(doc)), None)
    result.data.update(source=source, signature=signature, vector_pages=vector_pages, raster_pages=len(raster_pages),
                       codecs=codecs, recognised_text_chars=recognised, printed_text_chars=printed)

    if producer:
        result.add("producer", f"Made by {producer[1]} (\"{who}\").", "info")
    elif who:
        result.add("producer", f"Made by \"{who}\".", "info")
    else:
        result.add("producer", "The file names no program that made it.", "info")
    if camera:
        result.add("camera", f"Photo metadata names a camera: {camera}.", "info")

    if source == "vector-ink":
        extra = " with a handwriting-recognition text layer, as note apps add" if recognised else ""
        result.add("vector-ink", f"The ink is stored as vector pen strokes{extra}: how tablet note apps export.", "info")
        score = 0.05
    elif source == "camera-or-scan":
        result.add("captured", "The pages look photographed or scanned (sensor noise, uneven light or JPEG).", "info")
        score = 0.1
    elif source == "rendered-raster":
        result.add("rendered", "Each page is a single clean image: uniform paper and no camera or scanner noise. Software drew "
                   "it, or a scanner cleaned it (a renderer, a screenshot, an app that flattens pages, or a scanner that "
                   "whitens the background).", "weak")
        score = 0.35 if producer and producer[0] in ("library", "converter") else 0.25
    else:
        score = 0.25
    if signature:
        label = SIGNATURES[signature][0]
        result.add("signature", f"The file has the exact layout of {label}: pdf-lib, one lossless full-page image per page "
                   "at 2× or 4× a 612×792 page.", "strong")
        score = max(score, 0.85)
    result.score = score
    return result
