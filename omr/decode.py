from __future__ import annotations

import numpy as np

from .heads import detect_heads
from .key_signature import apply_alteration
from .segment import Component
from .staff import StaffMetrics, _column_runs, staff_line_rows

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


_BEAM_DURATION = {0: "quarter", 1: "eighth", 2: "sixteenth",
                  3: "thirty_second", 4: "sixty_fourth"}


def _find_stem(nostaff: np.ndarray, head: Component, metrics: StaffMetrics):
    h, w = nostaff.shape
    x0 = max(0, head.x1 - metrics.line_height)
    x1 = min(w, head.x2 + metrics.line_height)
    min_len = int(1.5 * metrics.staff_unit)
    best = (0, None, None, None)
    for x in range(x0, x1):
        for s, ln, val in zip(*_column_runs(nostaff[:, x])):
            if val and ln > best[0]:
                best = (ln, x, s, s + ln)
    if best[0] < min_len:
        return None
    return best[1], best[2], best[3]


def _count_beams(nostaff: np.ndarray, head: Component, stem, metrics: StaffMetrics) -> int:
    stem_x, top, bottom = stem
    head_cy = (head.y1 + head.y2) / 2
    u = metrics.staff_unit
    h, w = nostaff.shape
    if top < head_cy:
        y0, y1 = max(0, top - metrics.line_height), min(h, top + 2 * u)
    else:
        y0, y1 = max(0, bottom - 2 * u), min(h, bottom + metrics.line_height)

    reach = int(1.6 * u)
    left = nostaff[y0:y1, max(0, stem_x - reach):stem_x].sum()
    right = nostaff[y0:y1, stem_x + 1:stem_x + 1 + reach].sum()
    if max(left, right) == 0:
        return 0
    if right >= left:
        cols = range(stem_x + 2, min(w, stem_x + reach))
    else:
        cols = range(max(0, stem_x - reach), stem_x - 1)

    max_beam = 1.4 * u
    counts = []
    for x in cols:
        runs = sum(1 for s, ln, val in zip(*_column_runs(nostaff[y0:y1, x]))
                   if val and metrics.line_height <= ln <= max_beam)
        if runs:
            counts.append(runs)
    if not counts:
        return 0
    counts = np.array(counts)
    beams = 1
    for k in (2, 3, 4):
        if np.mean(counts >= k) >= 0.3:
            beams = k
    return beams


def head_duration(nostaff: np.ndarray, head: Component, metrics: StaffMetrics) -> str:
    crop = nostaff[head.y1:head.y2, head.x1:head.x2]
    hollow = crop.sum() / max(1, crop.size) < 0.55
    stem = _find_stem(nostaff, head, metrics)

    if stem is None:
        return "whole" if hollow else "quarter"
    if hollow:
        return "half"
    return _BEAM_DURATION.get(_count_beams(nostaff, head, stem, metrics),
                              "sixty_fourth")


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
           clef_token: str, heads: list[Component] | None = None,
           alt_map: dict[str, str] | None = None) -> list[dict]:
    if heads is None:
        heads = detect_heads(nostaff, metrics)
    heads = clean_heads(heads)
    bottom_y, half = staff_geometry(mask, metrics)
    v_ref, d_ref = clef_reference(clef_token)

    notes = []
    for h in sorted(heads, key=lambda c: c.x1):
        pitch = head_pitch(h, bottom_y, half, v_ref, d_ref)
        notes.append({
            "pitch": apply_alteration(pitch, alt_map) if alt_map else pitch,
            "duration": head_duration(nostaff, h, metrics),
        })
    return notes
