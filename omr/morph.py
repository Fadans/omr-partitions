from __future__ import annotations

import numpy as np


def disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def rect(height: int, width: int) -> np.ndarray:
    return np.ones((height, width), dtype=bool)


def erode(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    sh, sw = se.shape
    cy, cx = sh // 2, sw // 2
    h, w = mask.shape
    padded = np.pad(mask, ((cy, sh - cy - 1), (cx, sw - cx - 1)),
                    mode="constant", constant_values=False)
    out = np.ones((h, w), dtype=bool)
    for dy, dx in np.argwhere(se):
        out &= padded[dy:dy + h, dx:dx + w]
    return out


def dilate(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    sh, sw = se.shape
    cy, cx = sh // 2, sw // 2
    h, w = mask.shape
    padded = np.pad(mask, ((cy, sh - cy - 1), (cx, sw - cx - 1)),
                    mode="constant", constant_values=False)
    out = np.zeros((h, w), dtype=bool)
    for dy, dx in np.argwhere(se):
        out |= padded[dy:dy + h, dx:dx + w]
    return out


def opening(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return dilate(erode(mask, se), se)


def closing(mask: np.ndarray, se: np.ndarray) -> np.ndarray:
    return erode(dilate(mask, se), se)
