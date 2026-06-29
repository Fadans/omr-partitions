from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Component:
    label: int
    x1: int
    y1: int
    x2: int
    y2: int
    area: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def label_components(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: list[int] = [0]

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    next_label = 1
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            left = labels[y, x - 1] if x > 0 else 0
            up = labels[y - 1, x] if y > 0 else 0
            ul = labels[y - 1, x - 1] if (x > 0 and y > 0) else 0
            ur = labels[y - 1, x + 1] if (y > 0 and x + 1 < w) else 0
            neigh = [n for n in (left, up, ul, ur) if n]
            if not neigh:
                labels[y, x] = next_label
                parent.append(next_label)
                next_label += 1
            else:
                m = min(neigh)
                labels[y, x] = m
                for n in neigh:
                    union(m, n)

    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    nz = np.flatnonzero(labels)
    flat = labels.ravel()
    for idx in nz:
        root = find(flat[idx])
        if root not in remap:
            remap[root] = len(remap) + 1
        out.ravel()[idx] = remap[root]
    return out


def components_from_labels(labels: np.ndarray) -> list[Component]:
    n = int(labels.max())
    if n == 0:
        return []
    out: list[Component] = []
    for lab in range(1, n + 1):
        ys, xs = np.where(labels == lab)
        if xs.size == 0:
            continue
        out.append(Component(
            label=lab,
            x1=int(xs.min()), y1=int(ys.min()),
            x2=int(xs.max()) + 1, y2=int(ys.max()) + 1,
            area=int(xs.size),
        ))
    return out


def segment(mask: np.ndarray, min_area: int = 8) -> list[Component]:
    labels = label_components(mask)
    comps = [c for c in components_from_labels(labels) if c.area >= min_area]
    comps.sort(key=lambda c: c.x1)
    return comps
