from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from omr.dataset import prepared
from omr.decode import (_count_beams, _find_stem, beam_mask, clean_heads,
                        head_duration)
from omr.descriptors import projection_features
from omr.heads import detect_heads
from omr.io_primus import list_samples, parse_semantic
from omr.knn import NearestNeighbor

DURS = ["whole", "half", "quarter", "eighth", "sixteenth", "thirty_second"]


def base_dur(d: str) -> str:
    return d.split(".")[0]


def region_descriptor(nostaff, bmask, head, metrics):
    u = metrics.staff_unit
    stem = _find_stem(nostaff, head, metrics)
    crop_head = nostaff[head.y1:head.y2, head.x1:head.x2]
    hollow = crop_head.sum() / max(1, crop_head.size) < 0.55
    h, w = nostaff.shape
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
