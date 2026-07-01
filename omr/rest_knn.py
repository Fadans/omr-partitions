from __future__ import annotations

import os

import numpy as np

from .descriptors import hu_moments, projection_features
from .knn import NearestNeighbor
from .rests import detect_rests
from .segment import Component
from .staff import StaffMetrics, staff_line_rows

REST_DURS = ("whole", "half", "quarter", "eighth", "sixteenth", "thirty_second")


def _base(d: str) -> str:
    return d.split(".")[0]


def rest_descriptor(nostaff: np.ndarray, c: Component, metrics: StaffMetrics,
                    s_top: float, s_bot: float) -> np.ndarray:
    # descripteur de forme + position verticale (whole/half se distinguent par
    # leur appui sur la ligne). Même esprit que duration_knn (MLRF L1/L3).
    u = metrics.staff_unit
    crop = nostaff[c.y1:c.y2, c.x1:c.x2]
    fill = crop.sum() / max(1, crop.size)
    cy = (c.y1 + c.y2) / 2
    rel = (cy - s_top) / max(1.0, (s_bot - s_top))
    return np.concatenate([
        projection_features(crop, bins=12),
        hu_moments(crop),
        [c.width / u, c.height / u, fill, c.width / max(1, c.height), rel],
    ])


def _mid_rest_tokens(toks):
    types = [t.split("-")[0] for t in toks]
    idx = [i for i, t in enumerate(types) if t in ("note", "gracenote")]
    if not idx:
        return []
    lo, hi = idx[0], idx[-1]
    return [toks[i] for i in range(lo, hi) if types[i] == "rest"]


class RestClassifier:
    """Classe une composante candidate en durée de silence ou 'none'.

    Décision à haute précision : on n'accepte un silence que si les k plus
    proches voisins sont unanimes (sinon 'none'), pour limiter les faux
    silences qui pollueraient la sortie.
    """

    def __init__(self, k: int = 5, unanimous: bool = True):
        self.knn = NearestNeighbor(k=k)
        self.unanimous = unanimous

    def fit_features(self, X, y) -> "RestClassifier":
        self.knn.fit(X, y)
        return self

    def fit_samples(self, samples, prepared_fn) -> "RestClassifier":
        from .decode import clean_heads
        from .heads import detect_heads
        X, y = [], []
        for s in samples:
            with open(s.semantic, encoding="utf-8") as f:
                toks = f.read().split()
            mask, nostaff, metrics, comps = prepared_fn(s)
            rows = staff_line_rows(mask)
            if len(rows) < 2:
                continue
            s_top, s_bot = min(rows), max(rows)
            heads = clean_heads(detect_heads(nostaff, metrics))
            cand = sorted(detect_rests(comps, heads, metrics), key=lambda c: c.x1)
            gt = _mid_rest_tokens(toks)
            if len(gt) == 0:
                labels = ["none"] * len(cand)
            elif len(cand) == len(gt):
                labels = [_base(t.split("-")[-1]) for t in gt]
            else:
                continue
            for c, lab in zip(cand, labels):
                X.append(rest_descriptor(nostaff, c, metrics, s_top, s_bot))
                y.append(lab if lab in REST_DURS else "none")
        return self.fit_features(X, y)

    def _predict_vec(self, v) -> str:
        Xs = self.knn._scale(np.atleast_2d(v))[0]
        d = np.linalg.norm(self.knn.X - Xs, axis=1)
        idx = np.argsort(d)[:self.knn.k]
        labels = [self.knn.y[i] for i in idx]
        if self.unanimous and len(set(labels)) > 1:
            return "none"
        return max(set(labels), key=labels.count)

    def detect(self, nostaff, comps, heads, metrics, s_top, s_bot):
        """Retourne [(x, duration)] des silences détectés, triés par x."""
        out = []
        for c in sorted(detect_rests(comps, heads, metrics), key=lambda c: c.x1):
            lab = self._predict_vec(rest_descriptor(nostaff, c, metrics, s_top, s_bot))
            if lab != "none":
                out.append(((c.x1 + c.x2) / 2, lab))
        return out

    def save(self, path: str) -> None:
        np.savez(path, X=self.knn.X, mean=self.knn.mean, std=self.knn.std,
                 y=np.array(self.knn.y), k=self.knn.k, unanimous=self.unanimous)

    @classmethod
    def load(cls, path: str) -> "RestClassifier":
        d = np.load(path, allow_pickle=True)
        c = cls(k=int(d["k"]), unanimous=bool(d["unanimous"]))
        c.knn.X = d["X"]; c.knn.mean = d["mean"]; c.knn.std = d["std"]
        c.knn.y = list(d["y"])
        return c


DEFAULT_MODEL = os.path.join(os.path.dirname(__file__), os.pardir,
                             "models", "rest_knn.npz")


def load_default() -> "RestClassifier | None":
    path = os.path.abspath(DEFAULT_MODEL)
    return RestClassifier.load(path) if os.path.exists(path) else None
