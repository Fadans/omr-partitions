from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .morph import opening, rect


@dataclass
class StaffMetrics:
    line_height: int
    line_space: int

    @property
    def staff_unit(self) -> int:
        return self.line_height + self.line_space


def _column_runs(col: np.ndarray):
    h = col.shape[0]
    change = np.flatnonzero(np.diff(col.astype(np.int8)))
    bounds = np.concatenate(([-1], change, [h - 1]))
    lengths = np.diff(bounds)
    values = col[bounds[1:]]
    starts = bounds[:-1] + 1
    return starts, lengths, values


def estimate_metrics(mask: np.ndarray) -> StaffMetrics:
    h, w = mask.shape
    black = np.zeros(h + 1, dtype=np.int64)
    white = np.zeros(h + 1, dtype=np.int64)
    for x in range(w):
        _, lengths, values = _column_runs(mask[:, x])
        for ln, val in zip(lengths, values):
            (black if val else white)[ln] += 1
    line_height = int(np.argmax(black[1:]) + 1)
    line_space = int(np.argmax(white[1:]) + 1)
    return StaffMetrics(line_height=line_height, line_space=line_space)


def staff_line_rows(mask: np.ndarray, ratio: float = 0.5) -> list[int]:
    proj = mask.sum(axis=1)
    if proj.max() == 0:
        return []
    on = proj >= ratio * proj.max()
    rows: list[int] = []
    start = None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        elif not v and start is not None:
            rows.append((start + y - 1) // 2)
            start = None
    if start is not None:
        rows.append((start + len(on) - 1) // 2)
    return rows


def staff_lines_mask(mask: np.ndarray, metrics: StaffMetrics) -> np.ndarray:
    length = max(3, 2 * metrics.line_space)
    return opening(mask, rect(1, length))


def remove_staff_lines(mask: np.ndarray, metrics: StaffMetrics) -> np.ndarray:
    height = 2 * metrics.line_height + 1
    return opening(mask, rect(height, 1))


def remove_staff_lines_combined(mask: np.ndarray, metrics: StaffMetrics,
                                rows: list[int] | None = None) -> np.ndarray:
    if rows is None:
        rows = staff_line_rows(mask)
    return remove_staff_lines(mask, metrics) | remove_staff_lines_bridge(mask, metrics, rows)


def remove_staff_lines_bridge(mask: np.ndarray, metrics: StaffMetrics,
                              rows: list[int] | None = None) -> np.ndarray:
    if rows is None:
        rows = staff_line_rows(mask)
    if len(rows) < 2:
        return remove_staff_lines(mask, metrics)
    out = mask.copy()
    th = metrics.line_height
    gap = 2
    h, w = mask.shape
    for r in rows:
        r0, r1 = max(0, r - th), min(h, r + th + 1)
        a0, b1 = max(0, r0 - gap), min(h, r1 + gap)
        band = mask[r0:r1]
        above = mask[a0:r0].any(axis=0) if r0 > a0 else np.zeros(w, bool)
        below = mask[r1:b1].any(axis=0) if b1 > r1 else np.zeros(w, bool)
        bridged = above & below
        cols = band.any(axis=0) & ~bridged
        out[r0:r1, cols] = False
    return out


def remove_staff_lines_targeted(mask: np.ndarray, metrics: StaffMetrics,
                                rows: list[int] | None = None) -> np.ndarray:
    if rows is None:
        rows = staff_line_rows(mask)
    if len(rows) < 2:
        return remove_staff_lines(mask, metrics)
    out = mask.copy()
    tol = max(metrics.line_height, round(0.4 * metrics.line_space))
    max_run = 2 * metrics.line_height
    rows_arr = np.array(rows)
    h, w = mask.shape
    for x in range(w):
        starts, lengths, values = _column_runs(mask[:, x])
        for s, ln, val in zip(starts, lengths, values):
            if val and ln <= max_run:
                cy = s + ln / 2
                if np.abs(rows_arr - cy).min() <= tol:
                    out[s:s + ln, x] = False
    return out


def _stable_path(mask: np.ndarray, r0: int, band: int,
                 move_penalty: float = 0.4) -> np.ndarray:
    # plus court chemin gauche->droite suivant l'encre, dans une bande autour de
    # r0 (chemins stables, Cardoso 2009) : coût 0 sur l'encre, 1 sur le fond,
    # pénalité de déplacement vertical pour préférer une ligne lisse.
    h, w = mask.shape
    lo, hi = max(0, r0 - band), min(h, r0 + band + 1)
    cost = np.where(mask[lo:hi, :], 0.0, 1.0)
    R = cost.shape[0]
    dp = np.full((R, w), np.inf)
    back = np.zeros((R, w), dtype=np.int8)
    dp[:, 0] = cost[:, 0]
    for c in range(1, w):
        prev = dp[:, c - 1]
        up = np.full(R, np.inf); up[1:] = prev[:-1] + move_penalty   # depuis i-1
        down = np.full(R, np.inf); down[:-1] = prev[1:] + move_penalty  # depuis i+1
        stack = np.vstack([up, prev, down])
        choice = np.argmin(stack, axis=0).astype(np.int8)
        dp[:, c] = cost[:, c] + np.min(stack, axis=0)
        back[:, c] = choice
    path = np.empty(w, dtype=np.int64)
    i = int(np.argmin(dp[:, w - 1]))
    for c in range(w - 1, -1, -1):
        path[c] = lo + i
        ch = back[i, c]
        if ch == 0:
            i = max(0, i - 1)
        elif ch == 2:
            i = min(R - 1, i + 1)
    return path


def remove_staff_lines_stable_paths(mask: np.ndarray, metrics: StaffMetrics,
                                    rows: list[int] | None = None) -> np.ndarray:
    if rows is None:
        rows = staff_line_rows(mask)
    if len(rows) < 2:
        return remove_staff_lines(mask, metrics)
    out = mask.copy()
    band = max(metrics.line_height, round(0.4 * metrics.line_space))
    max_run = 2 * metrics.line_height
    h, w = mask.shape
    for r0 in rows:
        path = _stable_path(mask, r0, band)
        for c in range(w):
            rp = int(path[c])
            if not mask[rp, c]:
                continue
            top = rp
            while top - 1 >= 0 and mask[top - 1, c]:
                top -= 1
            bot = rp
            while bot + 1 < h and mask[bot + 1, c]:
                bot += 1
            if bot - top + 1 <= max_run:  # trait fin = ligne ; sinon symbole -> pont
                out[top:bot + 1, c] = False
    return out


def remove_staff_lines_runlength(mask: np.ndarray, metrics: StaffMetrics) -> np.ndarray:
    out = mask.copy()
    threshold = 2 * metrics.line_height
    h, w = mask.shape
    for x in range(w):
        starts, lengths, values = _column_runs(mask[:, x])
        for s, ln, val in zip(starts, lengths, values):
            if val and ln <= threshold:
                out[s:s + ln, x] = False
    return out
