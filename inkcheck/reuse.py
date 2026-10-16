"""Layer 2: the same shape written twice.

People never write a letter the same way twice; software that assembles
handwriting from samples does, nudged by a small turn, slant or stretch.
This layer finds pieces of ink with a twin elsewhere in the document that
lines up too well once that nudge is undone, and asks whether there are more
of them than real handwriting ever shows.

Steps: ink pieces → keep the complex ones (long, curved strokes; a dot or a
straight bar repeats naturally) → likely twins by a cheap descriptor → ECC
alignment and the stroke distance after it (register.py) → sort what lines
up (sort_twins): printed type (twins whose changes fit one smooth page
distortion, or a line of rescaled-only repeats, and whose pieces the
printed-type model reads as type), copied blocks (several twins moved by one
shared map: a writer's own copy and paste), and everything else is a reused
shape → a binomial test of the reused-shape count against the rate measured
on real pages.
"""

from __future__ import annotations

import copy
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from importlib import resources

import numpy as np
from scipy.stats import binom

from .ink import Component, InkMap, components, extract_ink
from .layers import LayerResult
from .load import Document
from .register import align, transform_parts
from .shapes import ShapeInfo, describe


def load_calibration() -> dict:
    with resources.files("inkcheck").joinpath("calibration.json").open() as f:
        return json.load(f)


@dataclass
class Piece:
    comp: Component
    info: ShapeInfo
    stroke_width: float
    letter_height: float
    typeset: float | None = None  # the printed-type model's answer, once asked


@dataclass
class Match:
    a: int
    b: int
    similarity: float
    displacement: float
    correlation: float
    matrix: np.ndarray
    parts: dict
    kind: str = "reused"  # reused | printed | block
    group: int = -1       # which copied block (or repeated printed word) it belongs to


@dataclass
class ReuseFindings:
    pieces: list[Piece]
    complex_count: int
    twins: list[Match] = field(default_factory=list)


def complexity(info: ShapeInfo) -> float:
    """Centreline length in pen widths, discounted to nothing for straight strokes."""
    return info.length_sw * float(np.clip((info.tortuosity - 1.0) / 0.3, 0.0, 1.0))


def collect(doc: Document, inks: list[InkMap] | None = None, minimum: float | None = None) -> tuple[list[Piece], list[InkMap]]:
    cal = load_calibration()
    minimum = cal["complexity_min"] if minimum is None else minimum
    inks = inks or [extract_ink(p.pixels, p.dpi) for p in doc.pages]
    pieces = []
    for page, ink in zip(doc.pages, inks):
        for c in components(ink, page.index, page.facts.text_boxes):
            if c.printed:
                continue
            info = describe(c, ink.stroke_width)
            info.complexity = complexity(info)
            if info.complexity >= minimum:
                pieces.append(Piece(c, info, ink.stroke_width, ink.letter_height))
    return pieces, inks


def candidate_pairs(pieces: list[Piece], k: int = 5, min_similarity: float = 0.75) -> list[tuple[int, int, float]]:
    if len(pieces) < 2:
        return []
    d = np.stack([p.info.descriptor for p in pieces])
    area = np.array([p.comp.area / max(p.stroke_width, 1e-6) ** 2 for p in pieces])
    aspect = np.array([p.info.aspect for p in pieces])
    pairs: dict[tuple[int, int], float] = {}
    top = min(k * 3, len(pieces) - 1)
    for start in range(0, len(pieces), 2048):  # in blocks, so a long document stays small in memory
        block = d[start : start + 2048] @ d.T
        for row, i in enumerate(range(start, min(start + 2048, len(pieces)))):
            block[row, i] = -1
        nearest = np.argpartition(-block, top - 1, axis=1)[:, :top] if top < len(pieces) else np.argsort(-block, axis=1)
        for row, i in enumerate(range(start, min(start + 2048, len(pieces)))):
            order = nearest[row][np.argsort(-block[row, nearest[row]])]
            _take(pairs, i, order, block[row], area, aspect, k, min_similarity)
    return [(a, b, s) for (a, b), s in pairs.items()]


def _take(pairs, i, order, sims, area, aspect, k, min_similarity) -> None:
    """Up to k look-alikes of piece i, of similar size and shape, into `pairs`."""
    taken = 0
    for j in order:
        if sims[j] < min_similarity or taken >= k:
            break
        if not (0.6 <= area[i] / area[j] <= 1.67 and 0.7 <= aspect[i] / aspect[j] <= 1.43):
            continue
        pairs[(min(i, j), max(i, j))] = float(sims[j])
        taken += 1


def _verify_chunk(items) -> list[tuple]:
    import cv2

    cv2.setNumThreads(0)  # one thread per worker process; the pool supplies the parallelism
    out = []
    for a, b, sim, ca, cb, sw in items:
        r = align(ca, cb, sw, early_exit=0.97)
        if r.ok:
            out.append((a, b, sim, r.displacement, r.correlation, r.matrix))
    return out


def verify(pieces: list[Piece], pairs, workers: int | None = None) -> list[Match]:
    # The distance is read in b's frame, so it is measured in b's own pen widths.
    items = [(a, b, s, pieces[a].comp, pieces[b].comp, pieces[b].stroke_width) for a, b, s in pairs]
    if not items:
        return []
    workers = workers if workers is not None else min(8, os.cpu_count() or 1)
    if workers > 1 and len(items) > 200:
        chunks = [items[i::workers * 4] for i in range(workers * 4)]
        with ProcessPoolExecutor(workers) as pool:
            rows = [row for part in pool.map(_verify_chunk, chunks) for row in part]
    else:
        rows = _verify_chunk(items)
    return [Match(a, b, sim, d, corr, m, transform_parts(m)) for a, b, sim, d, corr, m in rows]


def _rigid(parts: dict, tol: float = 0.02, turn_deg: float = 0.6) -> bool:
    """Same size, angle and slant, within what alignment can measure."""
    return (abs(parts["scale"] - 1) < tol and abs(parts["rotation_deg"]) < turn_deg and abs(parts["shear"]) < tol
            and abs(parts["aspect"] - 1) < tol)


def _shared_maps(pieces: list[Piece], twins: list[Match]) -> np.ndarray:
    """same[i, j]: twin j moves its piece the way twin i's map would (linear parts alike, and i's
    map sends j's first piece within two pen widths of j's second). Comparing translations
    directly would not do: they are measured at the page's corner, so a small difference in
    the linear part grows with distance from it."""
    L = np.array([m.matrix[:, :2] for m in twins])
    t = np.array([m.matrix[:, 2] for m in twins])
    ca = np.array([pieces[m.a].comp.centroid for m in twins])
    cb = np.array([pieces[m.b].comp.centroid for m in twins])
    pages = np.array([(pieces[m.a].comp.page, pieces[m.b].comp.page) for m in twins])
    reach = 2.0 * np.array([pieces[m.b].stroke_width for m in twins])
    alike = np.abs(L[:, None] - L[None]).reshape(len(twins), len(twins), 4).max(axis=2) < 0.012
    predicted = np.einsum("ijk,lk->ilj", L, ca) + t[:, None, :]          # i's map applied to j's first piece
    landed = np.linalg.norm(predicted - cb[None], axis=2) < reach[None]
    same_pages = (pages[:, None] == pages[None]).all(axis=2)
    return alike & landed & same_pages


def sort_twins(pieces: list[Piece], twins: list[Match]) -> None:
    """Marks printed type and copied blocks among the twins, in place.

    1. Candidate blocks: three or more twins carried by one shared map (a
       note app's lasso copy moves a whole group of letters this way; so does
       a word printed twice).
    2. Printed type, judged on the twins outside those groups: a line whose
       twins nearly all repeat at most rescaled (printed_lines), and twins
       whose shape changes are all explained by one smooth distortion of the
       page, as a flat scan or a photo's perspective gives every printed
       letter (printed_by_field).
    3. A candidate block whose letters are mostly printed is a repeated
       printed word; any other is a copied block of the writer's own ink."""
    groups: list[list[Match]] = []
    grouped: set[int] = set()
    by_pages: dict[tuple[int, int], list[Match]] = {}
    for m in twins:
        by_pages.setdefault((pieces[m.a].comp.page, pieces[m.b].comp.page), []).append(m)
    for ms in by_pages.values():
        if len(ms) < 3:
            continue
        same = _shared_maps(pieces, ms)
        for i, m in enumerate(ms):
            if id(m) in grouped:
                continue
            members = [ms[j] for j in np.flatnonzero(same[i]) if id(ms[j]) not in grouped]
            if len(members) >= 3:
                groups.append(members)
                grouped |= {id(n) for n in members}
    loose = [m for m in twins if id(m) not in grouped]
    printed = printed_clusters(pieces, loose) | printed_lines(pieces, loose) | printed_by_field(pieces, loose)
    printed |= _rest_of_printed_lines(pieces, loose, printed)
    printed |= _fragments_amid_print(pieces, loose, printed)
    for m in loose:
        if m.a in printed or m.b in printed:
            m.kind = "printed"
    for number, group in enumerate(groups):
        members = {i for n in group for i in (n.a, n.b)}
        by_shape = _looks_typeset(pieces, members) if _has_print_model() else False
        kind = "printed" if len(members & printed) >= 0.5 * len(members) or by_shape else "block"
        for n in group:
            n.kind = kind
            n.group = number


def _rest_of_printed_lines(pieces: list[Piece], twins: list[Match], printed: set[int]) -> set[int]:
    """Twinned pieces on a line whose twinned pieces are mostly print already.

    A scan breaks some printed letters into fragments (half an e, the stem
    of a d) that the shape model cannot call on their own; a line of type is
    type throughout."""
    if not printed:
        return set()
    twinned = {i for m in twins for i in (m.a, m.b)}
    out: set[int] = set()
    for line in text_lines(pieces):
        on_line = [i for i in line if i in twinned]
        if len(on_line) >= 3 and sum(i in printed for i in on_line) >= 0.6 * len(on_line):
            out |= set(on_line)
    return out


def _has_print_model() -> bool:
    from .printed import load_model

    return load_model() is not None


def _fragments_amid_print(pieces: list[Piece], twins: list[Match], printed: set[int], near: float = 3.0,
                          neighbours: int = 3) -> set[int]:
    """Rigid twins surrounded by print are print.

    A scan breaks some printed letters into pieces (the bowl of a g below the
    line, a stray serif) that the shape model cannot call on their own and that
    sit off the line the rest of the type is on. Found on the CVL forms: all
    four false alarms there were the bowls of printed g's. A twin counts here
    only when it is rigid (same size, angle and slant) and each of its pieces
    has at least `neighbours` printed pieces within `near` letter heights, so
    a page's few misjudged pieces can never excuse handwriting elsewhere."""
    if len(printed) < neighbours:
        return set()
    centres = {i: np.array(pieces[i].comp.centroid) for i in printed}
    by_page: dict[int, np.ndarray] = {}
    for i in printed:
        by_page.setdefault(pieces[i].comp.page, []).append(centres[i])
    by_page = {k: np.array(v) for k, v in by_page.items()}

    def amid(i: int) -> bool:
        page = pieces[i].comp.page
        if page not in by_page:
            return False
        reach = near * max(pieces[i].letter_height, 1.0)
        d = np.linalg.norm(by_page[page] - np.array(pieces[i].comp.centroid), axis=1)
        return int(np.count_nonzero(d < reach)) >= neighbours

    out: set[int] = set()
    for m in twins:
        if m.a in printed and m.b in printed:
            continue
        # Fragments are small, so their measured turn and size wobble more than whole letters' do.
        if _rigid(m.parts, tol=0.035, turn_deg=1.0) and amid(m.a) and amid(m.b):
            out |= {m.a, m.b}
    return out


def printed_clusters(pieces: list[Piece], twins: list[Match]) -> set[int]:
    """Pieces in groups of twins whose median piece the printed-type model reads as type.

    On a real scan the turn and size measured between printed letters wobble
    (a round letter's turn is barely defined), so geometry alone misses type;
    the shape model decides here, a group at a time."""
    from .printed import load_model

    if load_model() is None:
        return set()
    out: set[int] = set()
    for group in _components_of(twins):
        if _looks_typeset(pieces, group):
            out |= group
    return out


def _components_of(twins: list[Match]) -> list[set[int]]:
    parent: dict[int, int] = {}

    def find(i):
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for m in twins:
        parent[find(m.a)] = find(m.b)
    groups: dict[int, set[int]] = {}
    for i in list(parent):
        groups.setdefault(find(i), set()).add(i)
    return list(groups.values())


def printed_by_field(pieces: list[Piece], twins: list[Match], tolerance: float = 0.02, share: float = 0.6,
                     rounds: int = 300, seed: int = 0) -> set[int]:
    """Pieces whose twins' shape changes fit one smooth page distortion.

    For each twin on one page, the change in shape (its linear map with the
    overall size taken out, minus the identity) should be, for type, what a
    single smooth distortion of the page predicts between the two places:
    zero on a flat scan, a gentle gradient across a photographed page. The
    gradient (d = dx·Gx + dy·Gy) is fitted robustly (RANSAC); when it explains
    most of a page's twins, those twins are type. A generator's twins each
    carry their own random turn and slant, and no one field explains them."""
    rng = np.random.default_rng(seed)
    out: set[int] = set()
    # Twins are fitted page pair by page pair: a scanned or photographed page
    # carries its own small turn, so type repeated on two pages differs by one
    # constant change (c below) plus each page's gentle gradient.
    by_pages: dict[tuple[int, int], list[Match]] = {}
    for m in twins:
        key = tuple(sorted((pieces[m.a].comp.page, pieces[m.b].comp.page)))
        by_pages.setdefault(key, []).append(m)
    for (p, q), ms in by_pages.items():
        rows, used = [], []
        for m in ms:
            a, b = (m.a, m.b) if pieces[m.a].comp.page == p else (m.b, m.a)
            linear = m.matrix[:, :2] if a == m.a else np.linalg.inv(m.matrix[:, :2])
            det = float(np.linalg.det(linear))
            if det <= 0:
                continue
            d = (linear / np.sqrt(det) - np.eye(2)).ravel()
            (ax, ay), (bx, by) = pieces[a].comp.centroid, pieces[b].comp.centroid
            unit = max(pieces[a].letter_height, 1.0) * 40  # positions in units of 40 letter heights
            rows.append((1.0, (bx - ax) / unit, (by - ay) / unit, d))
            used.append(m)
        if len(used) < 3:
            continue
        constant = p != q
        X = np.array([[r[0], r[1], r[2]] if constant else [r[1], r[2]] for r in rows])
        D = np.array([r[3] for r in rows])
        best = np.linalg.norm(D, axis=1) < tolerance  # no distortion at all
        k = X.shape[1]
        for _ in range(rounds):
            pick = rng.choice(len(used), size=min(k + 1, len(used)), replace=False)
            G, *_ = np.linalg.lstsq(X[pick], D[pick], rcond=None)
            inliers = np.linalg.norm(D - X @ G, axis=1) < tolerance
            if inliers.sum() > best.sum():
                best = inliers
        if best.sum() >= max(3, share * len(used)):
            members = {i for m, ok in zip(used, best) if ok for i in (m.a, m.b)}
            if _looks_typeset(pieces, members):
                out |= members
    return out


def _looks_typeset(pieces: list[Piece], members: set[int]) -> bool:
    """Most of these pieces are shaped like type, by the printed-type model's median.

    Letters a library hand copies whole repeat rigidly too, so how twins
    repeat is not enough; their shapes are handwriting. With no model
    installed the answer is yes, and only the geometry decides."""
    from .printed import typeset_probability

    p = typeset_probability(pieces, members)
    return float(np.median(list(p.values()))) >= 0.5 if p else False


def text_lines(pieces: list[Piece]) -> list[list[int]]:
    """Pieces grouped into lines: neighbours whose heights overlap by half or more."""
    lines = []
    by_page: dict[int, list[int]] = {}
    for i, p in enumerate(pieces):
        by_page.setdefault(p.comp.page, []).append(i)
    for idx in by_page.values():
        boxes = np.array([pieces[i].comp.box for i in idx], float)
        x0, y0, w, h = boxes.T
        lh = np.median([pieces[i].letter_height for i in idx])
        overlap = np.minimum(y0[:, None] + h[:, None], (y0 + h)[None]) - np.maximum(y0[:, None], y0[None])
        near = np.abs((x0 + w / 2)[:, None] - (x0 + w / 2)[None]) < 6 * lh
        linked = (overlap >= 0.5 * np.minimum(h[:, None], h[None])) & near
        parent = list(range(len(idx)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b in zip(*np.nonzero(np.triu(linked, 1))):
            parent[find(a)] = find(b)
        groups: dict[int, list[int]] = {}
        for k in range(len(idx)):
            groups.setdefault(find(k), []).append(idx[k])
        lines.extend(groups.values())
    return lines


def baseline_spread(pieces: list[Piece], line: list[int]) -> float:
    """How far the line's letters stray from one straight baseline, in letter heights (descenders set aside)."""
    lh = float(np.median([pieces[i].letter_height for i in line]))
    x = np.array([pieces[i].comp.box[0] + pieces[i].comp.box[2] / 2 for i in line], float)
    y = np.array([pieces[i].comp.box[1] + pieces[i].comp.box[3] for i in line], float)
    keep = np.ones(len(line), bool)
    for _ in range(3):
        if keep.sum() < 4:
            return np.inf
        fit = np.polyfit(x[keep], y[keep], 1)
        residual = y - np.polyval(fit, x)
        keep = np.abs(residual - np.median(residual[keep])) < 0.12 * lh
    if keep.sum() < 4:
        return np.inf
    r = residual[keep]
    return float(np.median(np.abs(r - np.median(r))) / lh)


def _scale_only(parts: dict) -> bool:
    """Bigger or smaller at most: no turn, slant or squash. Type set at two sizes
    repeats this way; a hand-placed sample always picks up a turn or slant."""
    return abs(parts["rotation_deg"]) < 0.5 and abs(parts["shear"]) < 0.02 and abs(parts["aspect"] - 1) < 0.02


def printed_lines(pieces: list[Piece], twins: list[Match], share: float = 0.85) -> set[int]:
    """Pieces on lines of type: three or more twins touch the line, and nearly
    all of them repeat at most rescaled (or a lone twin repeats rigidly on a
    dead-straight baseline)."""
    printed: set[int] = set()
    for line in text_lines(pieces):
        if len(line) < 3:
            continue
        members = set(line)
        touching = [m for m in twins if m.a in members or m.b in members]
        plain = sum(_scale_only(m.parts) for m in touching)
        lone = (len(touching) == 1 and _rigid(touching[0].parts) and baseline_spread(pieces, line) < 0.008)
        if ((len(touching) >= 3 and plain >= share * len(touching)) or lone) and _looks_typeset(pieces, members):
            printed |= members
    return printed


def clusters(twins: list[Match], kind: str = "reused") -> list[list[int]]:
    parent: dict[int, int] = {}

    def find(i):
        while parent.setdefault(i, i) != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for m in twins:
        if m.kind == kind:
            parent[find(m.a)] = find(m.b)
    groups: dict[int, list[int]] = {}
    for i in parent:
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def find_twins(doc: Document, inks: list[InkMap] | None = None, workers: int | None = None,
               threshold: float | None = None) -> tuple[ReuseFindings, list[InkMap]]:
    cal = load_calibration()
    threshold = cal["twin_displacement"] if threshold is None else threshold
    pieces, inks = collect(doc, inks)
    pairs = candidate_pairs(pieces)
    matches = verify(pieces, pairs, workers)
    twins = [m for m in matches if m.displacement < threshold]
    sort_twins(pieces, twins)
    return ReuseFindings(pieces, len(pieces), twins), inks


def at_threshold(findings: ReuseFindings, threshold: float) -> ReuseFindings:
    """The same pieces with only the twins closer than `threshold`, sorted afresh."""
    twins = [copy.copy(m) for m in findings.twins if m.displacement < threshold]
    for m in twins:
        m.kind, m.group = "reused", -1
    sort_twins(findings.pieces, twins)
    return ReuseFindings(findings.pieces, findings.complex_count, twins)


def assess_reuse(doc: Document, inks: list[InkMap] | None = None, workers: int | None = None,
                 keep_findings: bool = False):
    """The reuse layer's result; with `keep_findings`, also the pieces and twins behind it."""
    if doc.error or not doc.pages:
        result = LayerResult("reuse", None)
        result.data.update(complex=0, duplicated=0, share=0.0, p_value=1.0, pen_width_px=0.0, printed=0, copied_blocks=0,
                           twin_pairs=0, too_little_reason="unreadable")
        result.add("unreadable", doc.error or "No pages.")
        return (result, None) if keep_findings else result
    findings, inks = find_twins(doc, inks, workers)
    result = layer_from(findings, inks)
    return (result, findings) if keep_findings else result


def layer_from(findings: ReuseFindings, inks: list[InkMap]) -> LayerResult:
    """The reuse layer's result from twins already found and sorted."""
    cal = load_calibration()
    result = LayerResult("reuse", None)
    pieces, twins = findings.pieces, findings.twins
    widths = [ink.stroke_width for ink in inks if ink.mask.any()]
    pen = float(np.median(widths)) if widths else 0.0
    reused = [m for m in twins if m.kind == "reused"]
    with_twin = sorted({i for m in reused for i in (m.a, m.b)})
    n, k = findings.complex_count, len(with_twin)
    copies = extra_copies(twins)
    p_value = float(binom.sf(copies - 1, n, cal["null_rate"])) if n and copies else 1.0
    result.data.update(complex=n, duplicated=k, extra_copies=copies, share=(k / n if n else 0.0), p_value=p_value, pen_width_px=pen,
                       printed=len({i for m in twins if m.kind == "printed" for i in (m.a, m.b)}),
                       copied_blocks=len({m.group for m in twins if m.kind == "block"}), twin_pairs=len(reused),
                       threshold=cal["twin_displacement"], null_rate=cal["null_rate"], too_little_reason=None)

    if pen and pen < cal["min_pen_px"]:
        result.data["too_little_reason"] = "resolution"
        result.add("resolution", f"The pen strokes are only {pen:.1f} px wide: too low a resolution to compare letters.")
        return result
    if n < cal["min_complex"]:
        result.data["too_little_reason"] = "few-shapes"
        result.add("few-shapes", f"Only {n} letters are complex enough to compare (at least {cal['min_complex']} are needed).")
        return result

    for group in clusters(twins, "reused")[:40]:
        result.evidence.append(_cluster_evidence(pieces, twins, group, "reused"))
    for number in sorted({m.group for m in twins if m.kind == "block"})[:5]:
        block = [m for m in twins if m.kind == "block" and m.group == number]
        members = sorted({i for m in block for i in (m.a, m.b)})
        result.evidence.append(_cluster_evidence(pieces, block, members, "block"))
    for group in clusters(twins, "printed")[:3]:
        result.evidence.append(_cluster_evidence(pieces, twins, group, "printed"))

    strength = min(1.0, max(0.0, -np.log10(max(p_value, 1e-300)) / 12.0))
    result.score = float(strength) if copies >= cal["min_copies"] else float(min(strength, 0.3))
    if k:
        groups = clusters(twins, "reused")
        result.add("reused", f"{k} of {n} complex letters have a twin elsewhere in the document that matches it to within "
                   f"{cal['twin_displacement']:.2f} pen widths once a small turn, slant or stretch is undone: "
                   f"{len(groups)} distinct shapes, some repeated up to {max(map(len, groups))} times, so {copies} extra "
                   f"copies. The test allows real handwriting up to {cal['null_rate'] * 1000:.1f} chance copies per 1,000 "
                   f"complex letters, far more than real pages showed (chance of this many or more: {_chance(p_value)}).",
                   "strong" if result.score >= 0.5 else "weak")
    else:
        result.add("no-reuse", f"None of the {n} complex letters has a twin: each one differs from its look-alikes, as handwriting does.")
    if result.data["copied_blocks"]:
        blocks = result.data["copied_blocks"]
        result.add("block", f"{blocks} group{'s' if blocks > 1 else ''} of ink appear{'' if blocks > 1 else 's'} twice, moved as a whole: "
                   "copied and pasted (a note app's lasso copy, for example), which writers do with their own work. "
                   "Not counted as reuse.", "info")
    if result.data["printed"]:
        result.add("printed", f"{result.data['printed']} letters repeat the way printed type does (the same shape, changed only "
                   "by the page's own tilt or perspective): printed text, left out of the test.")
    return result


def extra_copies(twins: list[Match]) -> int:
    """Pieces explained as copies of another: each group of copies counts its size minus one.

    Twinned pieces come in pairs, so counting pieces would count one chance
    coincidence twice and the binomial test, which assumes independent
    letters, would overstate its evidence."""
    return sum(len(group) - 1 for group in clusters(twins, "reused"))


def _chance(p: float) -> str:
    return "below 1e-300" if p <= 0 else f"{p:.1e}"


def _cluster_evidence(pieces: list[Piece], twins: list[Match], group: list[int], kind: str) -> dict:
    members = sorted(group, key=lambda i: (pieces[i].comp.page, pieces[i].comp.box[1], pieces[i].comp.box[0]))
    in_group = set(group)
    pairs = [m for m in twins if m.kind == kind and m.a in in_group and m.b in in_group]
    pairs.sort(key=lambda m: m.displacement)
    return {
        "kind": kind,
        "members": [{"page": pieces[i].comp.page, "box": list(map(int, pieces[i].comp.box)), "piece": i} for i in members],
        "pairs": [{"a": m.a, "b": m.b, "displacement": round(m.displacement, 4), "correlation": round(m.correlation, 5),
                   "matrix": np.round(m.matrix, 5).tolist(),
                   "turn_deg": round(m.parts["rotation_deg"], 2), "scale": round(m.parts["scale"], 4),
                   "shear": round(m.parts["shear"], 4)} for m in pairs[:6]],
    }
