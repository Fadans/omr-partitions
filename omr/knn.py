from __future__ import annotations

import numpy as np


class NearestNeighbor:
    def __init__(self, k: int = 1):
        self.k = k
        self.X: np.ndarray | None = None
        self.y: list = []
        self.mean = None
        self.std = None

    def fit(self, X, y) -> "NearestNeighbor":
        X = np.asarray(X, dtype=np.float64)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std == 0] = 1.0
        self.X = (X - self.mean) / self.std
        self.y = list(y)
        return self

    def _scale(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mean) / self.std

    def predict(self, X) -> list:
        Xs = self._scale(np.atleast_2d(X))
        out = []
        for v in Xs:
            d = np.linalg.norm(self.X - v, axis=1)
            idx = np.argsort(d)[:self.k]
            labels = [self.y[i] for i in idx]
            out.append(max(set(labels), key=labels.count))
        return out
