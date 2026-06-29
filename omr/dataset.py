from __future__ import annotations

import numpy as np

from .descriptors import describe
from .io_primus import PrimusSample, load_gray
from .preprocess import preprocess
from .segment import Component, segment
from .staff import StaffMetrics, estimate_metrics, remove_staff_lines


def prepared(sample: PrimusSample, distorted: bool = False):
    path = sample.distorted if distorted else sample.png
    mask = preprocess(load_gray(path), adaptive=distorted)
    metrics = estimate_metrics(mask)
    nostaff = remove_staff_lines(mask, metrics)
    comps = segment(nostaff, min_area=metrics.line_height * metrics.line_space)
    return nostaff, metrics, comps


def leftmost_tall(comps: list[Component], metrics: StaffMetrics) -> Component | None:
    tall = [c for c in comps if c.height >= 2 * metrics.line_space]
    if not tall:
        return None
    return min(tall, key=lambda c: c.x1)


def clef_label(sample: PrimusSample) -> str:
    with open(sample.semantic, encoding="utf-8") as f:
        return f.read().split()[0]


def clef_descriptor(sample: PrimusSample, distorted: bool = False):
    nostaff, metrics, comps = prepared(sample, distorted)
    clef = leftmost_tall(comps, metrics)
    if clef is None:
        return None
    return describe(nostaff, clef)
