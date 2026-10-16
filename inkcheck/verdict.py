"""Putting the layers together into one answer a person can act on.

Only the reuse layer can say "generated": it is the one layer whose evidence
survives printing, scanning and rewritten metadata, and whose evidence a
person can check by eye. A known generator's export layout and the ink
texture can make a page worth a closer look ("suspicious"), never more; a
page that is merely a clean image is not suspicious (scanners that whiten
the background make those too). Copied blocks (a
writer pasting their own ink) are reported but never raise the verdict.
Nothing here ever says a page is human-written.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .layers import LayerResult

LABELS = ("generated", "suspicious", "no-evidence", "too-little", "error")


@dataclass
class Verdict:
    label: str
    headline: str
    reasons: list[str] = field(default_factory=list)


def judge(layers: dict[str, LayerResult], error: str | None = None) -> Verdict:
    if error:
        return Verdict("error", f"Could not read the file: {error}")
    reuse = layers.get("reuse")
    provenance = layers.get("provenance")
    texture = layers.get("texture")
    reasons: list[str] = []

    if reuse and reuse.score is not None and reuse.score >= 0.5:
        d = reuse.data
        reasons.append(_finding(reuse, "reused"))
        if provenance and provenance.data.get("signature"):
            reasons.append(_finding(provenance, "signature"))
        return Verdict("generated", f"Assembled from samples: {d['duplicated']} of {d['complex']} complex letters are copies "
                       "of one another, which handwriting never does.", [r for r in reasons if r])

    suspicious = []
    if provenance and provenance.data.get("signature"):
        suspicious.append(_finding(provenance, "signature"))
    if reuse and reuse.score is not None and 0.2 <= reuse.score < 0.5:
        suspicious.append(_finding(reuse, "reused"))
    if texture and texture.score is not None and texture.score >= 0.8:
        suspicious.append(_finding(texture, "texture"))
    suspicious = [s for s in suspicious if s]

    too_little = reuse is not None and reuse.score is None
    if suspicious and (provenance and provenance.data.get("signature") or not too_little or len(suspicious) > 1):
        return Verdict("suspicious", "Worth a closer look; nothing here proves the handwriting was generated.", suspicious)
    if too_little:
        why = reuse.findings[-1].message if reuse.findings else "too little handwriting"
        return Verdict("too-little", "Not enough handwriting to judge. " + why, suspicious)
    if reuse:
        reasons.append(_finding(reuse, "no-reuse") or _finding(reuse, "reused"))
    return Verdict("no-evidence", "No sign of generated handwriting found.", [r for r in reasons if r])


def _finding(layer: LayerResult, code: str) -> str | None:
    for f in layer.findings:
        if f.code == code:
            return f.message
    return None


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text
