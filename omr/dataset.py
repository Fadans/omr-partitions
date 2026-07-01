from __future__ import annotations

import numpy as np

from .decode import clean_heads
from .descriptors import describe, projection_features
from .heads import detect_heads
from .io_primus import PrimusSample, load_gray
from .preprocess import preprocess
from .segment import Component, segment
from .staff import (
    StaffMetrics,
    estimate_metrics,
    remove_staff_lines_combined,
    staff_line_rows,
)


def prepared(sample: PrimusSample, distorted: bool = False):
    path = sample.distorted if distorted else sample.png
    mask = preprocess(load_gray(path), adaptive=distorted)
    metrics = estimate_metrics(mask)
    nostaff = remove_staff_lines_combined(mask, metrics)
    comps = segment(nostaff, min_area=metrics.line_height * metrics.line_space)
    return mask, nostaff, metrics, comps


def leftmost_tall(comps: list[Component], metrics: StaffMetrics) -> Component | None:
    # clef = composante haute ET assez large (exclut barres de mesure et
    # résidus de portée, fins, qui sont hauts mais très étroits).
    tall = [c for c in comps if c.height >= 2 * metrics.line_space
            and c.width >= 0.5 * metrics.staff_unit]
    if not tall:
        return None
    return min(tall, key=lambda c: c.x1)


def clef_label(sample: PrimusSample) -> str:
    with open(sample.semantic, encoding="utf-8") as f:
        return f.read().split()[0]


def clef_descriptor(sample: PrimusSample, distorted: bool = False):
    _, nostaff, metrics, comps = prepared(sample, distorted)
    clef = leftmost_tall(comps, metrics)
    if clef is None:
        return None
    return describe(nostaff, clef)


def _column_clusters(strip, thr_ratio: float = 0.15) -> int:
    col = strip.sum(axis=0)
    on = col > thr_ratio * strip.shape[0]
    n = 0
    prev = False
    for v in on:
        if v and not prev:
            n += 1
        prev = v
    return n


def keysig_strip_descriptor(sample: PrimusSample, distorted: bool = False, bins: int = 16):
    mask, nostaff, metrics, comps = prepared(sample, distorted)
    clef = leftmost_tall(comps, metrics)
    heads = clean_heads(detect_heads(nostaff, metrics))
    rows = staff_line_rows(mask)
    if clef is None or not heads or len(rows) < 2:
        return None
    first_x = min(h.x1 for h in heads)
    if first_x - clef.x2 < metrics.line_space:
        return None
    pad = metrics.staff_unit
    y1 = max(0, min(rows) - pad)
    y2 = min(nostaff.shape[0], max(rows) + pad)
    strip = nostaff[y1:y2, clef.x2:first_x]
    su = metrics.staff_unit
    width = strip.shape[1] / su
    ink = strip.sum() / (su * su)
    clusters = _column_clusters(strip)
    return np.concatenate([
        projection_features(strip, bins=bins),
        [width, ink, clusters, clusters / max(1.0, width)],
    ])


def key_token(sample: PrimusSample) -> str:
    with open(sample.semantic, encoding="utf-8") as f:
        toks = f.read().split()
    return toks[1] if len(toks) > 1 and toks[1].startswith("keySignature") else "CM"
