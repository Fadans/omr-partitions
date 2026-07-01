from __future__ import annotations

import numpy as np

from .heads import detect_heads
from .key_signature import apply_alteration
from .morph import opening, rect
from .segment import Component, segment
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


def beam_mask(nostaff: np.ndarray, metrics: StaffMetrics) -> np.ndarray:
    # Ouverture morphologique horizontale (TIFO2 §5.3) : isole les traits
    # horizontaux longs = ligatures ; hampes fines et têtes rondes disparaissent.
    width = max(3, int(1.1 * metrics.staff_unit))
    return opening(nostaff, rect(1, width))


def _count_beams(beams_mask: np.ndarray, head: Component, stem,
                 metrics: StaffMetrics) -> int:
    stem_x, top, bottom = stem
    head_cy = (head.y1 + head.y2) / 2
    u = metrics.staff_unit
    h, w = beams_mask.shape
    if top < head_cy:
        y0, y1 = max(0, top - metrics.line_height), min(h, top + 3 * u)
    else:
        y0, y1 = max(0, bottom - 3 * u), min(h, bottom + metrics.line_height)

    # au niveau de la hampe, chaque ligature est un segment vertical distinct :
    # on compte les runs verticaux dans le masque de ligatures, autour de la hampe.
    best = 0
    for x in range(max(0, stem_x - 1), min(w, stem_x + 2)):
        runs = sum(1 for s, ln, val in zip(*_column_runs(beams_mask[y0:y1, x]))
                   if val and ln <= 1.4 * u)
        best = max(best, runs)
    return best


def head_duration(nostaff: np.ndarray, head: Component, metrics: StaffMetrics,
                  beams_mask: np.ndarray | None = None) -> str:
    crop = nostaff[head.y1:head.y2, head.x1:head.x2]
    hollow = crop.sum() / max(1, crop.size) < 0.55
    stem = _find_stem(nostaff, head, metrics)

    if stem is None:
        return "whole" if hollow else "quarter"
    if hollow:
        return "half"
    if beams_mask is None:
        beams_mask = beam_mask(nostaff, metrics)
    return _BEAM_DURATION.get(_count_beams(beams_mask, head, stem, metrics),
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
           alt_map: dict[str, str] | None = None,
           dur_clf=None, rest_clf=None) -> list[dict]:
    if heads is None:
        heads = detect_heads(nostaff, metrics)
    heads = clean_heads(heads)
    bottom_y, half = staff_geometry(mask, metrics)
    v_ref, d_ref = clef_reference(clef_token)
    bmask = beam_mask(nostaff, metrics)

    items = []  # (x, entrée) pour fusionner notes et silences dans l'ordre
    for h in sorted(heads, key=lambda c: c.x1):
        pitch = head_pitch(h, bottom_y, half, v_ref, d_ref)
        if dur_clf is not None:
            duration = dur_clf.predict_head(nostaff, bmask, h, metrics)
        else:
            duration = head_duration(nostaff, h, metrics, bmask)
        items.append(((h.x1 + h.x2) / 2, {
            "pitch": apply_alteration(pitch, alt_map) if alt_map else pitch,
            "duration": duration,
        }))

    if rest_clf is not None and heads:
        rows = staff_line_rows(mask)
        if len(rows) >= 2:
            comps = segment(nostaff, min_area=metrics.line_height * metrics.line_space)
            for x, dur in rest_clf.detect(nostaff, comps, heads, metrics,
                                          min(rows), max(rows)):
                items.append((x, {"pitch": "rest", "duration": dur}))

    items.sort(key=lambda t: t[0])
    return [entry for _, entry in items]
