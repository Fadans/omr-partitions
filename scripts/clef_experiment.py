from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from omr.dataset import clef_descriptor, clef_label
from omr.io_primus import list_samples
from omr.knn import NearestNeighbor


def build(samples, distorted):
    X, y = [], []
    for s in samples:
        d = clef_descriptor(s, distorted=distorted)
        if d is None:
            continue
        X.append(d)
        y.append(clef_label(s))
    return np.array(X), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--distorted", action="store_true")
    args = ap.parse_args()

    samples = list_samples(args.corpus, limit=args.n)
    split = int(0.7 * len(samples))
    Xtr, ytr = build(samples[:split], args.distorted)
    Xte, yte = build(samples[split:], args.distorted)

    clf = NearestNeighbor(k=args.k).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = np.mean([p == t for p, t in zip(pred, yte)])

    print(f"train={len(ytr)} test={len(yte)} classes={len(set(ytr))}")
    print(f"k={args.k} accuracy clef = {acc * 100:.1f} %")
    err = Counter((t, p) for t, p in zip(yte, pred) if t != p)
    if err:
        print("erreurs (vrai -> prédit):")
        for (t, p), n in err.most_common(8):
            print(f"  {t} -> {p}  x{n}")


if __name__ == "__main__":
    main()
