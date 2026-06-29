from __future__ import annotations

import numpy as np

from .segment import Component


def hu_moments(crop: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(crop)
    if xs.size == 0:
        return np.zeros(7)
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    m00 = float(xs.size)
    xb = xs.mean()
    yb = ys.mean()
    dx = xs - xb
    dy = ys - yb

    def mu(p, q):
        return float(np.sum(dx ** p * dy ** q))

    def eta(p, q):
        return mu(p, q) / m00 ** (1 + (p + q) / 2)

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n03, n21, n12 = eta(3, 0), eta(0, 3), eta(2, 1), eta(1, 2)

    h = np.empty(7)
    h[0] = n20 + n02
    h[1] = (n20 - n02) ** 2 + 4 * n11 ** 2
    h[2] = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
    h[3] = (n30 + n12) ** 2 + (n21 + n03) ** 2
    h[4] = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) \
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
    h[5] = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) \
        + 4 * n11 * (n30 + n12) * (n21 + n03)
    h[6] = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) \
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)

    return -np.sign(h) * np.log10(np.abs(h) + 1e-30)


def _resample(profile: np.ndarray, bins: int) -> np.ndarray:
    if profile.size == 0:
        return np.zeros(bins)
    xp = np.linspace(0.0, 1.0, profile.size)
    x = np.linspace(0.0, 1.0, bins)
    out = np.interp(x, xp, profile)
    s = out.sum()
    return out / s if s > 0 else out


def projection_features(crop: np.ndarray, bins: int = 8) -> np.ndarray:
    row = crop.sum(axis=1).astype(np.float64)
    col = crop.sum(axis=0).astype(np.float64)
    return np.concatenate([_resample(row, bins), _resample(col, bins)])


def crop_of(mask: np.ndarray, comp: Component) -> np.ndarray:
    return mask[comp.y1:comp.y2, comp.x1:comp.x2]


def shape_features(crop: np.ndarray, bins: int = 8) -> np.ndarray:
    h, w = crop.shape
    aspect = w / h if h else 0.0
    fill = crop.sum() / (h * w) if h and w else 0.0
    return np.concatenate([
        hu_moments(crop),
        projection_features(crop, bins),
        [aspect, fill],
    ])


def describe(mask: np.ndarray, comp: Component, bins: int = 8) -> np.ndarray:
    return shape_features(crop_of(mask, comp), bins)
