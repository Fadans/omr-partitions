import numpy as np

from omr.descriptors import hu_moments, projection_features, shape_features
from omr.knn import NearestNeighbor
from omr.morph import disk


def test_hu_translation_invariant():
    a = np.zeros((20, 20), dtype=bool)
    a[4:9, 5:12] = True
    b = np.zeros((20, 20), dtype=bool)
    b[10:15, 2:9] = True
    assert np.allclose(hu_moments(a), hu_moments(b), atol=1e-6)


def test_hu_distinguishes_shapes():
    square = np.ones((12, 12), dtype=bool)
    line = np.zeros((12, 12), dtype=bool)
    line[5:7, :] = True
    assert not np.allclose(hu_moments(square), hu_moments(line), atol=0.2)


def test_projection_features_normalised():
    crop = np.zeros((10, 10), dtype=bool)
    crop[2:8, 3:7] = True
    f = projection_features(crop, bins=8)
    assert f.shape == (16,)
    assert abs(f[:8].sum() - 1.0) < 1e-9
    assert abs(f[8:].sum() - 1.0) < 1e-9


def test_knn_classifies_disk_vs_bar():
    samples, labels = [], []
    for r in (4, 5, 6):
        samples.append(shape_features(disk(r)))
        labels.append("disk")
    for w in (10, 12, 14):
        bar = np.zeros((3, w), dtype=bool)
        bar[:] = True
        samples.append(shape_features(bar))
        labels.append("bar")
    clf = NearestNeighbor(k=1).fit(samples, labels)
    assert clf.predict(shape_features(disk(5)))[0] == "disk"
    test_bar = np.ones((3, 11), dtype=bool)
    assert clf.predict(shape_features(test_bar))[0] == "bar"
