import numpy as np

from omr.preprocess import binarize, median_blur, otsu_threshold


def test_otsu_separates_bimodal():
    img = np.full((20, 20), 50, dtype=np.uint8)
    img[:, 10:] = 200
    t = otsu_threshold(img)
    assert 50 <= t < 200
    mask = img <= t
    assert mask[:, :10].all()
    assert not mask[:, 10:].any()


def test_binarize_marks_ink_as_true():
    img = np.full((10, 10), 220, dtype=np.uint8)
    img[2:5, 2:5] = 10
    mask = binarize(img)
    assert mask[3, 3]
    assert not mask[0, 0]


def test_median_blur_removes_isolated_pixel():
    img = np.full((7, 7), 255, dtype=np.uint8)
    img[3, 3] = 0
    out = median_blur(img, 3)
    assert out[3, 3] == 255


def test_median_blur_preserves_shape():
    img = np.random.randint(0, 256, (15, 23), dtype=np.uint8)
    assert median_blur(img, 3).shape == img.shape
