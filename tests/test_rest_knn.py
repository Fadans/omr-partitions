import numpy as np

from omr.rest_knn import RestClassifier, rest_descriptor
from omr.segment import Component
from omr.staff import StaffMetrics


def _synthetic():
    metrics = StaffMetrics(line_height=2, line_space=8)  # staff_unit = 10
    img = np.zeros((80, 40), dtype=bool)
    img[36:44, 14:22] = True            # petit bloc centré (silence factice)
    comp = Component(1, 14, 36, 22, 44, 64)
    return img, metrics, comp


def test_rest_descriptor_shape():
    img, metrics, comp = _synthetic()
    d = rest_descriptor(img, comp, metrics, s_top=20.0, s_bot=60.0)
    assert d.shape == (36,)             # 2*12 projections + 7 hu + 5 features
    assert np.isfinite(d).all()


def test_rest_classifier_unanimous_and_roundtrip(tmp_path):
    img, metrics, comp = _synthetic()
    d = rest_descriptor(img, comp, metrics, 20.0, 60.0)
    # 5 voisins non unanimes -> 'none' (haute précision)
    X = [d, d + 0.01, d - 0.01, d + 0.02, d - 0.02]
    y = ["quarter", "quarter", "eighth", "none", "none"]
    clf = RestClassifier(k=5, unanimous=True).fit_features(X, y)
    assert clf._predict_vec(d) == "none"

    # unanimes -> la classe
    clf2 = RestClassifier(k=3, unanimous=True).fit_features(
        [d, d + 0.01, d - 0.01], ["quarter", "quarter", "quarter"])
    assert clf2._predict_vec(d) == "quarter"

    path = str(tmp_path / "rest.npz")
    clf.save(path)
    clf3 = RestClassifier.load(path)
    assert clf3.knn.k == 5 and clf3.unanimous is True
    assert clf3._predict_vec(d) == "none"
