from __future__ import annotations

import os

import numpy as np

from .decode import _count_beams, _find_stem, beam_mask, clean_heads
from .descriptors import projection_features
from .heads import detect_heads
from .knn import NearestNeighbor
from .segment import Component
from .staff import StaffMetrics


def base_duration(d: str) -> str:
    return d.split(".")[0]


def region_descriptor(nostaff: np.ndarray, bmask: np.ndarray, head: Component,
                      metrics: StaffMetrics) -> np.ndarray:
    # descripteur de forme (projections + géométrie) de la région tête+hampe+
    # ligature, pour classer la durée par k-NN (MLRF L1/L3).
    u = metrics.staff_unit
    h, w = nostaff.shape
    stem = _find_stem(nostaff, head, metrics)
    crop_head = nostaff[head.y1:head.y2, head.x1:head.x2]
    hollow = crop_head.sum() / max(1, crop_head.size) < 0.55
    if stem is not None:
        sx, top, bot = stem
        x0 = max(0, int(sx - 1.6 * u)); x1 = min(w, int(sx + 1.6 * u))
        y0 = max(0, int(top - 0.3 * u)); y1 = min(h, int(bot + 0.3 * u))
        stem_len = (bot - top) / u
        nb = _count_beams(bmask, head, stem, metrics)
    else:
        x0, y0, x1, y1 = head.x1, head.y1, head.x2, head.y2
        stem_len = 0.0
        nb = 0
    crop = nostaff[y0:y1, x0:x1]
    ch, cw = max(1, y1 - y0), max(1, x1 - x0)
    fill = crop.sum() / (ch * cw)
    aspect = cw / ch
    pf = projection_features(crop, bins=16)
    return np.concatenate([pf, [float(hollow), stem_len, fill, aspect, float(nb)]])


class DurationClassifier:
    def __init__(self, k: int = 5):
        self.knn = NearestNeighbor(k=k)
        self._fitted = False

    def fit_features(self, X, y) -> "DurationClassifier":
        self.knn.fit(X, y)
        self._fitted = True
        return self

    def fit_samples(self, samples, prepared_fn) -> "DurationClassifier":
        from .io_primus import parse_semantic
        X, y = [], []
        for s in samples:
            gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
            mask, nostaff, metrics, _ = prepared_fn(s)
            heads = clean_heads(detect_heads(nostaff, metrics))
            if len(heads) != len(gt):  # alignement fiable par ordre seulement
                continue
            bmask = beam_mask(nostaff, metrics)
            for head, g in zip(sorted(heads, key=lambda c: c.x1), gt):
                X.append(region_descriptor(nostaff, bmask, head, metrics))
                y.append(base_duration(g["duration"]))
        return self.fit_features(X, y)

    def predict_head(self, nostaff, bmask, head, metrics) -> str:
        return self.knn.predict(region_descriptor(nostaff, bmask, head, metrics))[0]

    def save(self, path: str) -> None:
        np.savez(path, X=self.knn.X, mean=self.knn.mean, std=self.knn.std,
                 y=np.array(self.knn.y), k=self.knn.k)

    @classmethod
    def load(cls, path: str) -> "DurationClassifier":
        d = np.load(path, allow_pickle=True)
        c = cls(k=int(d["k"]))
        c.knn.X = d["X"]; c.knn.mean = d["mean"]; c.knn.std = d["std"]
        c.knn.y = list(d["y"]); c._fitted = True
        return c


DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), os.pardir,
                             "models", "duration_knn.npz")


def load_default() -> "DurationClassifier | None":
    path = os.path.abspath(DEFAULT_MODEL)
    return DurationClassifier.load(path) if os.path.exists(path) else None
