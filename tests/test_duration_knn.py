import numpy as np

from omr.decode import beam_mask
from omr.duration_knn import DurationClassifier, base_duration, region_descriptor
from omr.segment import Component
from omr.staff import StaffMetrics


def _synthetic():
    # tête pleine + hampe verticale, sur fond vide (u=10, épaisseur 2)
    metrics = StaffMetrics(line_height=2, line_space=8)  # staff_unit = 10
    img = np.zeros((80, 40), dtype=bool)
    img[40:50, 10:20] = True          # tête pleine ~1u
    img[20:50, 18:20] = True          # hampe
    head = Component(1, 10, 40, 20, 50, 100)
    return img, metrics, head


def test_base_duration_strips_dot():
    assert base_duration("eighth.") == "eighth"
    assert base_duration("quarter") == "quarter"


def test_region_descriptor_shape():
    img, metrics, head = _synthetic()
    bmask = beam_mask(img, metrics)
    d = region_descriptor(img, bmask, head, metrics)
    assert d.shape == (37,)          # 2*16 projections + 5 features
    assert np.isfinite(d).all()


def test_classifier_fit_predict_and_roundtrip(tmp_path):
    img, metrics, head = _synthetic()
    bmask = beam_mask(img, metrics)
    d = region_descriptor(img, bmask, head, metrics)
    X = [d, d + 0.01, d - 0.01]
    y = ["quarter", "quarter", "eighth"]
    clf = DurationClassifier(k=1).fit_features(X, y)
    assert clf.predict_head(img, bmask, head, metrics) == "quarter"

    path = str(tmp_path / "dur.npz")
    clf.save(path)
    clf2 = DurationClassifier.load(path)
    assert clf2.predict_head(img, bmask, head, metrics) == "quarter"
