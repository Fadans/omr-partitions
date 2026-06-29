from __future__ import annotations

import numpy as np


def median_blur(img: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1:
        return img
    pad = k // 2
    p = np.pad(img, pad, mode="edge")
    h, w = img.shape
    stack = np.empty((k * k, h, w), dtype=img.dtype)
    n = 0
    for dy in range(k):
        for dx in range(k):
            stack[n] = p[dy:dy + h, dx:dx + w]
            n += 1
    return np.median(stack, axis=0).astype(img.dtype)


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    levels = np.arange(256)
    sum_total = float(np.dot(levels, hist))

    w_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    threshold = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var = var
            threshold = t
    return threshold


def _local_mean(gray: np.ndarray, window: int) -> np.ndarray:
    pad = window // 2
    p = np.pad(gray.astype(np.float64), pad, mode="reflect")
    cs = np.cumsum(np.cumsum(p, axis=0), axis=1)
    cs = np.pad(cs, ((1, 0), (1, 0)), mode="constant")
    h, w = gray.shape
    s = window
    total = cs[s:s + h, s:s + w] - cs[:h, s:s + w] - cs[s:s + h, :w] + cs[:h, :w]
    return total / (s * s)


def binarize(gray: np.ndarray, threshold: int | None = None,
             adaptive: bool = False, window: int = 25, c: float = 8.0) -> np.ndarray:
    if adaptive:
        return gray.astype(np.float64) <= (_local_mean(gray, window) - c)
    if threshold is None:
        threshold = otsu_threshold(gray)
    return gray <= threshold


def preprocess(gray: np.ndarray, median_k: int = 3,
               adaptive: bool = False) -> np.ndarray:
    smooth = median_blur(gray, median_k)
    return binarize(smooth, adaptive=adaptive)
