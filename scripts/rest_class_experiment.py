from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from omr.dataset import clean_heads, prepared
from omr.decode import beam_mask
from omr.descriptors import projection_features, hu_moments
from omr.heads import detect_heads
from omr.io_primus import list_samples, parse_semantic
from omr.knn import NearestNeighbor
from omr.rests import detect_rests
from omr.staff import staff_line_rows

REST_DURS = ["whole", "half", "quarter", "eighth", "sixteenth", "none"]


def base(d):
    return d.split(".")[0]


def rest_descriptor(nostaff, c, metrics, s_top, s_bot):
    u = metrics.staff_unit
    crop = nostaff[c.y1:c.y2, c.x1:c.x2]
    fill = crop.sum() / max(1, crop.size)
    cy = (c.y1 + c.y2) / 2
    rel = (cy - s_top) / max(1.0, (s_bot - s_top))  # position verticale relative
    return np.concatenate([
        projection_features(crop, bins=12),
        hu_moments(crop),
        [c.width / u, c.height / u, fill, c.width / max(1, c.height), rel],
    ])


def mid_rest_tokens(toks):
    types = [t.split("-")[0] for t in toks]
    idx = [i for i, t in enumerate(types) if t in ("note", "gracenote")]
    if not idx:
        return []
    lo, hi = idx[0], idx[-1]
    return [toks[i] for i in range(lo, hi) if types[i] == "rest"]


def collect(samples):
    X, y = [], []
    for s in samples:
        with open(s.semantic, encoding="utf-8") as f:
            toks = f.read().split()
        mask, nostaff, metrics, comps = prepared(s)
        rows = staff_line_rows(mask)
        if len(rows) < 2:
            continue
        s_top, s_bot = min(rows), max(rows)
        heads = clean_heads(detect_heads(nostaff, metrics))
        cand = sorted(detect_rests(comps, heads, metrics), key=lambda c: c.x1)
        gt = mid_rest_tokens(toks)
        if len(gt) == 0:
            labels = ["none"] * len(cand)
        elif len(cand) == len(gt):
            labels = [base(t.split("-")[-1]) for t in gt]
        else:
            continue
        for c, lab in zip(cand, labels):
            X.append(rest_descriptor(nostaff, c, metrics, s_top, s_bot))
            y.append(lab if lab in REST_DURS else "none")
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.7 * len(samples))

    Xtr, ytr = collect(samples[:split])
    Xte, yte = collect(samples[split:])
    print(f"train={len(ytr)}  {Counter(ytr)}")
    print(f"test={len(yte)}   {Counter(yte)}")
    clf = NearestNeighbor(k=args.k).fit(Xtr, ytr)
    pred = [clf.predict(x)[0] for x in Xte]

    # détection silence vs none
    tp = sum(1 for t, p in zip(yte, pred) if t != "none" and p != "none")
    fn = sum(1 for t, p in zip(yte, pred) if t != "none" and p == "none")
    fp = sum(1 for t, p in zip(yte, pred) if t == "none" and p != "none")
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    print(f"\ndétection silence : précision {100*prec:.1f}  rappel {100*rec:.1f}")
    # durée correcte parmi silences bien détectés
    dur_ok = sum(1 for t, p in zip(yte, pred) if t != "none" and t == p)
    real = sum(1 for t in yte if t != "none")
    print(f"durée correcte (sur tous les silences réels) : {100*dur_ok/max(1,real):.1f}")

    conf = {d: Counter() for d in REST_DURS}
    for t, p in zip(yte, pred):
        conf[t][p] += 1
    print("\nconfusion (row=GT):")
    print("  " + "".join(f"{c[:5]:>7}" for c in REST_DURS))
    for d in REST_DURS:
        r = conf[d]; rt = sum(r.values())
        if rt:
            print(f"  {d[:2]}" + "".join(f"{r.get(c,0):>7}" for c in REST_DURS)
                  + f"  {100*r.get(d,0)/rt:5.1f}")


if __name__ == "__main__":
    main()
