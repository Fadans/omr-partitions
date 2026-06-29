import numpy as np

from omr.heads import detect_heads, fill_holes
from omr.morph import disk
from omr.staff import StaffMetrics


def _filled_ellipse(h, w, cy, cx, ry, rx):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def test_fill_holes_closes_ring():
    mask = disk(6).copy()
    mask[6, 6] = False
    assert fill_holes(mask)[6, 6]


def test_detect_two_filled_heads_with_stems():
    metrics = StaffMetrics(line_height=3, line_space=15)
    mask = np.zeros((60, 120), dtype=bool)
    for cx in (30, 80):
        mask |= _filled_ellipse(60, 120, 30, cx, 7, 9)
        mask[5:30, cx + 8] = True            # hampe fine
    mask[40, :] = True                       # ligne résiduelle fine
    heads = detect_heads(mask, metrics)
    assert len(heads) == 2
    centers = sorted((h.x1 + h.x2) // 2 for h in heads)
    assert abs(centers[0] - 30) <= 4 and abs(centers[1] - 80) <= 4


def test_thin_stem_alone_not_detected():
    metrics = StaffMetrics(line_height=3, line_space=15)
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:50, 30] = True
    assert detect_heads(mask, metrics) == []
