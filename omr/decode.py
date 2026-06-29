from __future__ import annotations

import numpy as np

from .heads import detect_heads
from .segment import Component
from .staff import StaffMetrics, staff_line_rows

LETTERS = "CDEFGAB"
_CLEF_REF = {"G": ("G", 4), "C": ("C", 4), "F": ("F", 3)}


def _pitch_value(letter: str, octave: int) -> int:
    return 7 * octave + LETTERS.index(letter)


def _value_to_pitch(v: int) -> str:
    return f"{LETTERS[v % 7]}{v // 7}"


def clef_reference(clef_token: str) -> tuple[int, int]:
    body = clef_token.split("-")[1]
    letter, line = body[0], int(body[1])
    ref_letter, ref_octave = _CLEF_REF[letter]
    return _pitch_value(ref_letter, ref_octave), 2 * (line - 1)


def staff_geometry(mask: np.ndarray, metrics: StaffMetrics) -> tuple[float, float]:
    rows = staff_line_rows(mask)
    if len(rows) >= 2:
        bottom = float(max(rows))
        half = (max(rows) - min(rows)) / ((len(rows) - 1) * 2)
        return bottom, half
    bottom = float(mask.shape[0] - 1)
    return bottom, metrics.staff_unit / 2


def head_pitch(head: Component, bottom_y: float, half: float,
               v_ref: int, d_ref: int) -> str:
    cy = (head.y1 + head.y2) / 2
    d = round((bottom_y - cy) / half)
    return _value_to_pitch(v_ref + (d - d_ref))


def head_duration(nostaff: np.ndarray, head: Component, metrics: StaffMetrics) -> str:
    crop = nostaff[head.y1:head.y2, head.x1:head.x2]
    fill = crop.sum() / max(1, crop.size)
    hollow = fill < 0.55

    col_sums = nostaff[:, max(0, head.x1 - 1):head.x2 + 1].sum(axis=0)
    has_stem = col_sums.max() >= 2.5 * metrics.staff_unit

    if hollow and not has_stem:
        return "whole"
    if hollow:
        return "half"
    if not has_stem:
        return "quarter"
    return "quarter" if not _has_flag_or_beam(nostaff, head, metrics) else "eighth"


def _has_flag_or_beam(nostaff: np.ndarray, head: Component, metrics: StaffMetrics) -> bool:
    band = nostaff[max(0, head.y1 - 3 * metrics.staff_unit):head.y1,
                   head.x2:min(nostaff.shape[1], head.x2 + 2 * metrics.staff_unit)]
    return band.sum() > metrics.staff_unit * metrics.line_height


def clean_heads(heads: list[Component]) -> list[Component]:
    if not heads:
        return []
    med_h = float(np.median([h.height for h in heads]))
    kept = [h for h in heads if h.height >= 0.6 * med_h]
    kept.sort(key=lambda c: c.x1)

    out: list[Component] = []
    for h in kept:
        if out:
            prev = out[-1]
            overlap = min(prev.x2, h.x2) - max(prev.x1, h.x1)
            if overlap > 0.5 * min(prev.width, h.width):
                if h.area > prev.area:
                    out[-1] = h
                continue
        out.append(h)
    return out


def decode(mask: np.ndarray, nostaff: np.ndarray, metrics: StaffMetrics,
           clef_token: str, heads: list[Component] | None = None) -> list[dict]:
    if heads is None:
        heads = detect_heads(nostaff, metrics)
    heads = clean_heads(heads)
    bottom_y, half = staff_geometry(mask, metrics)
    v_ref, d_ref = clef_reference(clef_token)

    notes = []
    for h in sorted(heads, key=lambda c: c.x1):
        notes.append({
            "pitch": head_pitch(h, bottom_y, half, v_ref, d_ref),
            "duration": head_duration(nostaff, h, metrics),
        })
    return notes
