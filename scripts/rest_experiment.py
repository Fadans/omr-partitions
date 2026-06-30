from __future__ import annotations

import argparse

import numpy as np

from omr.dataset import prepared
from omr.decode import clean_heads
from omr.descriptors import describe
from omr.heads import detect_heads
from omr.io_primus import list_samples
from omr.knn import NearestNeighbor
from omr.rests import detect_rests


def mid_rests(toks):
    types = [t.split("-")[0] for t in toks]
    note_idx = [i for i, t in enumerate(types) if t in ("note", "gracenote")]
    if not note_idx:
        return []
    lo, hi = note_idx[0], note_idx[-1]
    return [toks[i] for i in range(lo, hi) if types[i] == "rest"]


def collect(samples):
    rows = []
    for s in samples:
        with open(s.semantic, encoding="utf-8") as f:
            toks = f.read().split()
        mask, nostaff, metrics, comps = prepared(s)
        heads = clean_heads(detect_heads(nostaff, metrics))
        cand = detect_rests(comps, heads, metrics)
        gt = mid_rests(toks)
        if len(gt) == 0:
            label = "notrest"
        elif len(cand) == len(gt):
            label = "rest"
        else:
            continue
        for c in cand:
            rows.append((describe(nostaff, c), label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.7 * len(samples))
    train = collect(samples[:split])
    test = collect(samples[split:])
    X = [d for d, _ in train]
    y = [l for _, l in train]
    clf = NearestNeighbor(k=args.k).fit(X, y)
    print(f"train={len(y)} (rest={y.count('rest')}, notrest={y.count('notrest')})")

    tp = fp = fn = tn = 0
    for d, lab in test:
        pred = clf.predict(d)[0]
        if lab == "rest" and pred == "rest":
            tp += 1
        elif lab == "rest":
            fn += 1
        elif pred == "rest":
            fp += 1
        else:
            tn += 1
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"test symboles: rest={tp+fn} notrest={tn+fp}")
    print(f"  précision rest = {100*prec:.1f} %  rappel rest = {100*rec:.1f} %")
    print(f"  faux positifs filtrés: {tn}/{tn+fp} négatifs rejetés")


if __name__ == "__main__":
    main()
