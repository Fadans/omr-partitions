from __future__ import annotations

import numpy as np

from .morph import disk, opening
from .segment import Component, components_from_labels, label_components
from .staff import StaffMetrics


def fill_holes(mask: np.ndarray, max_hole_area: int | None = None) -> np.ndarray:
    free = ~mask
    labels = label_components(free)
    border = set(labels[0, :]) | set(labels[-1, :])
    border |= set(labels[:, 0]) | set(labels[:, -1])
    border.discard(0)
    if max_hole_area is None:
        outside = np.isin(labels, list(border))
        return mask | (free & ~outside)

    out = mask.copy()
    for lab in range(1, int(labels.max()) + 1):
        if lab in border:
            continue
        comp = labels == lab
        if comp.sum() <= max_hole_area:
            out |= comp
    return out


def detect_heads(mask: np.ndarray, metrics: StaffMetrics,
                 radius_ratio: float = 0.40,
                 min_fill: float = 0.55,
                 aspect_range: tuple[float, float] = (0.55, 2.4)) -> list[Component]:
    radius = max(1, round(metrics.line_space * radius_ratio))
    solid = fill_holes(mask, max_hole_area=2 * metrics.line_space ** 2)
    blobs = opening(solid, disk(radius))
    comps = components_from_labels(label_components(blobs))

    min_dim = 0.5 * metrics.line_space
    max_dim = 2.2 * metrics.line_space
    heads: list[Component] = []
    for c in comps:
        if not (min_dim <= c.width <= max_dim and min_dim <= c.height <= max_dim):
            continue
        if not (aspect_range[0] <= c.width / c.height <= aspect_range[1]):
            continue
        if c.area / (c.width * c.height) < min_fill:
            continue
        heads.append(c)
    heads.sort(key=lambda c: c.x1)
    return heads
