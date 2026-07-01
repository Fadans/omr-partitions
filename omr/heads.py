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


def _ellipse(a: int, b: int) -> np.ndarray:
    yy, xx = np.ogrid[-b:b + 1, -a:a + 1]
    return (xx * xx) / (a * a) + (yy * yy) / (b * b) <= 1.0


def detect_heads_template(mask: np.ndarray, metrics: StaffMetrics,
                          thresh: float = 0.62) -> list[Component]:
    """Détection de têtes par corrélation de gabarit (état de l'art §4, MLRF L3).

    Recouvrement d'un gabarit elliptique glissé sur l'image (têtes bouchées) ;
    on garde les maxima locaux au-dessus d'un seuil (suppression des non-maxima).

    NOTE : expérience comparative (cf. `scripts/benchmark_heads.py`). Le gabarit
    plein corrèle avec toute zone pleine assez grande (ligatures, clef,
    jonctions) -> sur-détection ~2x même à seuil élevé. L'ouverture
    morphologique (`detect_heads`) reste supérieure et est utilisée par défaut.
    """
    solid = fill_holes(mask, max_hole_area=2 * metrics.line_space ** 2)
    a = max(2, round(0.50 * metrics.line_space))
    b = max(2, round(0.42 * metrics.line_space))
    templ = _ellipse(a, b)
    area = int(templ.sum())
    h, w = solid.shape
    sh, sw = templ.shape
    cy, cx = sh // 2, sw // 2
    padded = np.pad(solid, ((cy, sh - cy - 1), (cx, sw - cx - 1)))
    acc = np.zeros((h, w), dtype=np.int32)
    for dy, dx in np.argwhere(templ):
        acc += padded[dy:dy + h, dx:dx + w]
    score = acc.astype(np.float64) / area

    ys, xs = np.nonzero(score >= thresh)
    order = np.argsort(score[ys, xs])[::-1]
    radius = max(2, int(0.75 * metrics.line_space))
    taken: list[tuple[int, int]] = []
    heads: list[Component] = []
    for i in order:
        y, x = int(ys[i]), int(xs[i])
        if any(abs(x - tx) < radius and abs(y - ty) < radius for tx, ty in taken):
            continue
        taken.append((x, y))
        heads.append(Component(len(heads) + 1, x - a, y - b, x + a + 1, y + b + 1,
                               area))
    heads.sort(key=lambda c: c.x1)
    return heads


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
