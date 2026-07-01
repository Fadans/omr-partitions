import numpy as np

from omr.staff import (
    StaffMetrics,
    estimate_metrics,
    remove_staff_lines,
    remove_staff_lines_stable_paths,
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


def test_stable_paths_removes_line_keeps_symbols():
    mask, _ = _synthetic_staff()
    m = estimate_metrics(mask)
    out = remove_staff_lines_stable_paths(mask, m)
    assert out[:, 50].sum() > 20          # hampe conservée (traversée = pont)
    assert out[24:30, 46:52].any()        # tête conservée
    for r in [10, 18, 34, 42]:            # lignes effacées hors symboles
        assert not out[r, :45].any()


def test_remove_staff_lines_keeps_beam():
    m = StaffMetrics(line_height=2, line_space=12)
    mask = np.zeros((50, 60), dtype=bool)
    mask[10:12, :] = True            # ligne fine (épaisseur 2)
    mask[30:36, 5:55] = True         # ligature épaisse (6 px de haut)
    out = remove_staff_lines(mask, m)
    assert not out[10:12, :].any()   # ligne supprimée
    assert out[30:36, 5:55].sum() > 100  # ligature conservée
