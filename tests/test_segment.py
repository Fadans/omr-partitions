import numpy as np

from omr.segment import label_components, segment


def test_label_two_blobs_diagonal():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True
    mask[6:9, 6:9] = True
    labels = label_components(mask)
    assert labels.max() == 2


def test_eight_connectivity_joins_diagonal():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1, 1] = True
    mask[2, 2] = True
    labels = label_components(mask)
    assert labels.max() == 1


def test_segment_filters_small_and_sorts():
    mask = np.zeros((20, 40), dtype=bool)
    mask[2:12, 30:36] = True   # gros, à droite
    mask[5:9, 2:6] = True      # moyen, à gauche
    mask[0, 0] = True          # bruit isolé
    comps = segment(mask, min_area=4)
    assert len(comps) == 2
    assert comps[0].x1 < comps[1].x1     # trié gauche -> droite


def test_component_bbox_fields():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 3:7] = True
    c = segment(mask, min_area=1)[0]
    assert c.bbox == (3, 2, 7, 5)
    assert c.width == 4 and c.height == 3 and c.area == 12
