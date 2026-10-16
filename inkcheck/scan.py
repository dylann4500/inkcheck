"""One file in, every layer run once over shared ink, one verdict out."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ink import InkMap, extract_ink
from .layers import LayerResult
from .load import Document, load_document
from .provenance import assess_provenance
from .reuse import ReuseFindings, assess_reuse
from .verdict import Verdict, judge

ALL_LAYERS = ("provenance", "reuse", "texture")


@dataclass
class ScanResult:
    document: Document
    layers: dict[str, LayerResult]
    verdict: Verdict
    inks: list[InkMap] = field(default_factory=list)
    reuse: ReuseFindings | None = None  # pieces and twins, for drawing the evidence


def scan_document(doc: Document, layers=ALL_LAYERS, workers: int | None = None) -> ScanResult:
    results: dict[str, LayerResult] = {}
    if doc.error:
        return ScanResult(doc, results, judge(results, doc.error))
    inks = [extract_ink(p.pixels, p.dpi) for p in doc.pages]
    if "provenance" in layers:
        results["provenance"] = assess_provenance(doc)
    findings = None
    if "reuse" in layers:
        results["reuse"], findings = assess_reuse(doc, inks, workers, keep_findings=True)
    if "texture" in layers:
        from .texture import assess_texture

        results["texture"] = assess_texture(doc, inks)
    return ScanResult(doc, results, judge(results), inks, findings)


def scan(path: str | Path, layers=ALL_LAYERS, workers: int | None = None) -> ScanResult:
    return scan_document(load_document(path), layers, workers)
