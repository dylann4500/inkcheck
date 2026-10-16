"""What every evidence layer returns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Weight = Literal["info", "weak", "moderate", "strong"]


@dataclass
class Finding:
    code: str
    message: str
    weight: Weight = "info"


@dataclass
class LayerResult:
    name: str
    # How strongly this layer points to generated handwriting, 0..1; None when
    # it could not judge (too little ink, no model, an unreadable page).
    score: float | None
    findings: list[Finding] = field(default_factory=list)
    data: dict = field(default_factory=dict)
    # Things to show a person: reuse clusters, regions, crops (by page and box).
    evidence: list[dict] = field(default_factory=list)

    def add(self, code: str, message: str, weight: Weight = "info") -> None:
        self.findings.append(Finding(code, message, weight))
