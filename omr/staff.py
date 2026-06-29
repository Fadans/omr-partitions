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
