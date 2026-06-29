import numpy as np

from omr.staff import (
    estimate_metrics,
    remove_staff_lines,
    staff_line_rows,
)


def _synthetic_staff():
    h, w = 60, 80
    mask = np.zeros((h, w), dtype=bool)
    line_rows = [10, 18, 26, 34, 42]
    for r in line_rows:
        mask[r, :] = True
    mask[8:45, 50] = True          # une hampe verticale
    mask[24:30, 46:52] = True      # une tête de note
    return mask, line_rows


def test_estimate_metrics():
    mask, _ = _synthetic_staff()
    m = estimate_metrics(mask)
    assert m.line_height == 1
    assert m.line_space == 7


def test_staff_line_rows_finds_five():
    mask, line_rows = _synthetic_staff()
    rows = staff_line_rows(mask)
    assert len(rows) == 5
    for got, exp in zip(rows, line_rows):
        assert abs(got - exp) <= 1


def test_remove_staff_lines_keeps_stem_and_head():
    mask, _ = _synthetic_staff()
    m = estimate_metrics(mask)
    out = remove_staff_lines(mask, m)
    assert out[:, 50].sum() > 20          # hampe conservée
    assert out[24:30, 46:52].any()        # tête conservée
    for r in [10, 18, 34, 42]:            # lignes effacées (hors hampe/tête)
        assert not out[r, :45].any()
