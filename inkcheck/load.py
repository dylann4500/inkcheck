"""Opening a submission: a PDF or an image file, as pages of pixels plus the
facts about how each page was stored.

A PDF page that is one image covering the page (how Scribe, scanners and
"flatten" exports store a page) yields that image's own pixels, so nothing is
resampled before the analysis. Any other PDF page is rendered at
`render_dpi` (288 by default, Scribe's export resolution).
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pymupdf
from PIL import Image, ImageOps

DEFAULT_DPI = 288.0

# PDF image filters, as the codec a person would name.
CODECS = {
    "FlateDecode": "flate",
    "DCTDecode": "jpeg",
    "JPXDecode": "jpeg2000",
    "JBIG2Decode": "jbig2",
    "CCITTFaxDecode": "ccitt",
    "RunLengthDecode": "runlength",
    "LZWDecode": "lzw",
    "": "raw",
}


@dataclass
class PageFacts:
    """How a page was stored, before anything is read from its pixels."""

    images: int = 0
    image_codecs: list[str] = field(default_factory=list)
    image_px: tuple[int, int] | None = None  # (width, height) of the largest image
    full_page_image: bool = False
    vector_segments: int = 0
    visible_text_chars: int = 0
    invisible_text_chars: int = 0
    size_pt: tuple[float, float] | None = None
    # Visible text, as pixel boxes (x0, y0, x1, y1) in this page's pixels.
    text_boxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class Page:
    index: int
    pixels: np.ndarray  # H×W×3 uint8 RGB
    dpi: float
    facts: PageFacts
    dpi_known: bool = True


@dataclass
class Document:
    path: Path
    kind: Literal["pdf", "image"]
    pages: list[Page] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    error: str | None = None


def load_document(path: str | Path, render_dpi: float = DEFAULT_DPI) -> Document:
    """Every page of the PDF or image at `path`; failures go in `error`, never raise."""
    path = Path(path)
    try:
        head = path.open("rb").read(8)
    except OSError as exc:
        return Document(path, "image", error=f"Could not open the file: {exc.strerror or exc}")
    try:
        if head.startswith(b"%PDF"):
            return _load_pdf(path, render_dpi)
        return _load_image(path)
    except Exception as exc:  # a broken file is reported, not raised
        kind = "pdf" if head.startswith(b"%PDF") else "image"
        return Document(path, kind, error=f"Could not read the file: {exc}")


# ---------------------------------------------------------------------------
# PDF


def _load_pdf(path: Path, render_dpi: float) -> Document:
    doc = pymupdf.open(path)
    if doc.needs_pass:
        return Document(path, "pdf", error="The PDF is password-protected.")
    metadata = {k: v for k, v in (doc.metadata or {}).items() if v}
    xmp = doc.get_xml_metadata() or ""
    if xmp:
        metadata["xmp"] = xmp[:4000]
    result = Document(path, "pdf", metadata=metadata)
    for index in range(doc.page_count):
        result.pages.append(_pdf_page(doc, doc[index], index, render_dpi))
    if not result.pages:
        result.error = "The PDF has no pages."
    return result


def _pdf_page(doc, page, index: int, render_dpi: float) -> Page:
    rect = page.rect
    facts = PageFacts(size_pt=(rect.width, rect.height))
    largest = None  # (area, xref, info tuple)
    covered = []
    for info in page.get_images(full=True):
        xref, width, height, filt = info[0], info[2], info[3], info[8]
        facts.images += 1
        codec = CODECS.get(filt, filt.lower() or "raw")
        if codec not in facts.image_codecs:
            facts.image_codecs.append(codec)
        for placed in page.get_image_rects(xref, transform=True):
            box, matrix = placed
            visible = box & rect
            covered.append((visible.get_area() / max(rect.get_area(), 1e-9), xref, matrix))
        if largest is None or width * height > largest[0]:
            largest = (width * height, xref, (width, height))
    if largest:
        facts.image_px = largest[2]
    facts.vector_segments = sum(len(d["items"]) for d in page.get_drawings())
    scale = render_dpi / 72.0
    for span in page.get_texttrace():
        chars = len(span.get("chars", ()))
        if span.get("type") == 3 or span.get("opacity", 1) == 0:
            facts.invisible_text_chars += chars
        else:
            facts.visible_text_chars += chars
            # The trace is in the page's unrotated space; rendered pixels follow its rotation.
            box = pymupdf.Rect(span["bbox"]) * page.rotation_matrix
            facts.text_boxes.append((box.x0 * scale, box.y0 * scale, box.x1 * scale, box.y1 * scale))

    full = [c for c in covered if c[0] >= 0.98]
    upright = bool(full) and abs(full[0][2].b) < 1e-6 and abs(full[0][2].c) < 1e-6 and full[0][2].a > 0 and full[0][2].d > 0
    if len(full) == 1 and facts.images == 1 and upright and facts.vector_segments < 50 and facts.visible_text_chars == 0:
        facts.full_page_image = True
        pixels = _pdf_image_pixels(doc, full[0][1])
        if pixels is not None:
            dpi = pixels.shape[1] / (rect.width / 72.0)
            facts.text_boxes = []
            return Page(index, pixels, float(dpi), facts)
    pix = page.get_pixmap(dpi=int(round(render_dpi)), alpha=False, colorspace=pymupdf.csRGB)
    pixels = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[..., :3].copy()
    return Page(index, pixels, float(render_dpi), facts)


def _pdf_image_pixels(doc, xref: int) -> np.ndarray | None:
    """An embedded image's own pixels as RGB on white (soft masks applied)."""
    try:
        pix = pymupdf.Pixmap(doc, xref)
        if pix.colorspace is None or pix.colorspace.n not in (1, 3):
            pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
        smask = doc.extract_image(xref).get("smask") or 0
        if smask:
            mask = pymupdf.Pixmap(doc, smask)
            pix = pymupdf.Pixmap(pix, mask)
        arr = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.alpha:
            alpha = arr[..., -1:].astype(np.float32) / 255
            colour = arr[..., :-1].astype(np.float32)
            if colour.shape[-1] == 1:
                colour = np.repeat(colour, 3, axis=-1)
            return np.clip(colour * alpha + 255 * (1 - alpha), 0, 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            return np.repeat(arr, 3, axis=-1).copy()
        return arr[..., :3].copy()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Image files


EXIF_TAGS = {271: "camera_make", 272: "camera_model", 305: "software", 306: "datetime", 42036: "lens_model"}


def _load_image(path: Path) -> Document:
    try:
        return _load_image_pil(path)
    except Exception:
        # Some TIFF variants open in Pillow but will not decode; OpenCV reads them.
        import cv2

        ok, frames = cv2.imreadmulti(str(path), flags=cv2.IMREAD_COLOR)
        if not ok or not frames:
            raise
        result = Document(path, "image", metadata={"format": path.suffix.upper().lstrip(".")})
        for index, bgr in enumerate(frames):
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            facts = PageFacts(images=1, image_codecs=[path.suffix.lower().lstrip(".") or "unknown"],
                              image_px=(rgb.shape[1], rgb.shape[0]), full_page_image=True)
            result.pages.append(Page(index, rgb, DEFAULT_DPI, facts, dpi_known=False))
        return result


def _load_image_pil(path: Path) -> Document:
    image = Image.open(path)
    metadata = {"format": image.format or ""}
    try:
        exif = image.getexif()
        for tag, name in EXIF_TAGS.items():
            if tag in exif and str(exif[tag]).strip():
                metadata[name] = str(exif[tag]).strip()
        sub = exif.get_ifd(0x8769) if exif else {}
        if 42036 in sub:
            metadata["lens_model"] = str(sub[42036])
    except Exception:
        pass
    result = Document(path, "image", metadata=metadata)
    # A phone's JPEG may carry a second image (a preview: format MPO); only the first is the page.
    frames = 1 if image.format == "MPO" else getattr(image, "n_frames", 1)
    for index in range(frames):
        if frames > 1:
            image.seek(index)
        frame = ImageOps.exif_transpose(image) if index == 0 else image
        rgb = np.asarray(_on_white(frame))
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            raise ValueError("could not decode the image")
        dpi_info = image.info.get("dpi")
        known = bool(dpi_info and dpi_info[0] and dpi_info[0] > 30)
        dpi = float(dpi_info[0]) if known else DEFAULT_DPI
        facts = PageFacts(images=1, image_codecs=[_image_codec(image.format)], image_px=(rgb.shape[1], rgb.shape[0]),
                          full_page_image=True)
        result.pages.append(Page(index, rgb.copy(), dpi, facts, dpi_known=known))
    return result


def _on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
        values = np.asarray(image, dtype=np.float64)
        top = 65535.0 if image.mode.startswith("I;16") else (1.0 if image.mode == "F" and values.max(initial=0) <= 1.0 else
                                                             max(float(values.max(initial=1)), 1.0))
        return Image.fromarray(np.clip(values / top * 255, 0, 255).astype(np.uint8)).convert("RGB")
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(white, rgba).convert("RGB")
    return image.convert("RGB")


def _image_codec(fmt: str | None) -> str:
    return {"PNG": "png", "JPEG": "jpeg", "MPO": "jpeg", "TIFF": "tiff", "HEIF": "heif", "WEBP": "webp", "BMP": "bmp", "GIF": "gif"}.get(fmt or "", (fmt or "unknown").lower())


def load_bytes(data: bytes, name: str = "upload", render_dpi: float = DEFAULT_DPI) -> Document:
    """The same, for bytes already in memory (the web page's uploads)."""
    import tempfile

    suffix = Path(name).suffix or (".pdf" if data[:4] == b"%PDF" else ".img")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        doc = load_document(tmp_path, render_dpi)
        doc.path = Path(name)
        return doc
    finally:
        tmp_path.unlink(missing_ok=True)
