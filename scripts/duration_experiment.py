from __future__ import annotations

import argparse
from collections import Counter

from omr.dataset import prepared
from omr.decode import beam_mask, clean_heads, head_duration
from omr.duration_knn import base_duration as base_dur, region_descriptor
from omr.heads import detect_heads
from omr.io_primus import list_samples, parse_semantic
from omr.knn import NearestNeighbor

DURS = ["whole", "half", "quarter", "eighth", "sixteenth", "thirty_second"]


def collect(samples):
    X, y = [], []
    for s in samples:
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        mask, nostaff, metrics, _ = prepared(s)
        heads = clean_heads(detect_heads(nostaff, metrics))
        if len(heads) != len(gt):
            continue  # aligne par ordre uniquement si comptage exact
        bmask = beam_mask(nostaff, metrics)
        heads = sorted(heads, key=lambda c: c.x1)
        for h, g in zip(heads, gt):
            X.append(region_descriptor(nostaff, bmask, h, metrics))
            y.append(base_dur(g["duration"]))
    return X, y


def geom_labels(samples):
    Xh, yg = [], []  # (descriptor, gt) plus geometric prediction
    preds = []
    for s in samples:
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        mask, nostaff, metrics, _ = prepared(s)
        heads = clean_heads(detect_heads(nostaff, metrics))
        if len(heads) != len(gt):
            continue
        bmask = beam_mask(nostaff, metrics)
        heads = sorted(heads, key=lambda c: c.x1)
        for h, g in zip(heads, gt):
            Xh.append(region_descriptor(nostaff, bmask, h, metrics))
            yg.append(base_dur(g["duration"]))
            preds.append(head_duration(nostaff, h, metrics, bmask))
    return Xh, yg, preds


def acc_report(name, y_true, y_pred):
    conf = {d: Counter() for d in DURS}
    ok = 0
    for t, p in zip(y_true, y_pred):
        if t in conf:
            conf[t][p] += 1
        ok += t == p
    print(f"\n{name}: overall {100*ok/len(y_true):.1f}%  (n={len(y_true)})")
    print("  " + "".join(f"{c[:5]:>7}" for c in DURS) + "   acc")
    for d in DURS:
        row = conf[d]; rt = sum(row.values())
        if not rt:
            continue
        cells = "".join(f"{row.get(c,0):>7}" for c in DURS)
        print(f"  {d[:2]}{cells}  {100*row.get(d,0)/rt:5.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.7 * len(samples))

    Xtr, ytr = collect(samples[:split])
    clf = NearestNeighbor(k=args.k).fit(Xtr, ytr)
    print(f"train notes={len(ytr)}  classes={Counter(ytr)}")

    Xte, yte, geom = geom_labels(samples[split:])
    knn = [clf.predict(x)[0] for x in Xte]

    acc_report("GEOMETRIC (actuel)", yte, geom)
    acc_report("kNN descripteurs", yte, knn)


if __name__ == "__main__":
    main()
