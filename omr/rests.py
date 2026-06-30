from __future__ import annotations

from .segment import Component
from .staff import StaffMetrics


def is_barline(c: Component, metrics: StaffMetrics) -> bool:
    return c.height >= 3 * metrics.staff_unit and c.width <= 2.5 * metrics.line_height


def detect_rests(comps: list[Component], heads: list[Component],
                 metrics: StaffMetrics) -> list[Component]:
    if not heads:
        return []
    x_lo = min(h.x1 for h in heads)
    x_hi = max(h.x2 for h in heads) + 2 * metrics.staff_unit
    head_spans = [(h.x1, h.x2) for h in heads]
    head_ids = {id(h) for h in heads}
    u = metrics.staff_unit

    rests = []
    for c in comps:
        if id(c) in head_ids:
            continue
        cx = (c.x1 + c.x2) / 2
        if not (x_lo <= cx <= x_hi):
            continue
        if is_barline(c, metrics):
            continue
        if any(not (c.x2 <= hx1 or c.x1 >= hx2) for hx1, hx2 in head_spans):
            continue
        if not (0.5 * u <= c.height <= 3.5 * u):
            continue
        if c.width > 2 * u:
            continue
        rests.append(c)
    return rests
