from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("livrable2_evaluation", "tools"))
from metrics import note_scores  # noqa: E402

from omr.dataset import prepared  # noqa: E402
from omr.decode import decode  # noqa: E402
from omr.duration_knn import DurationClassifier  # noqa: E402
from omr.io_primus import list_samples, parse_semantic  # noqa: E402


def evaluate(samples, dur_clf):
    geo = {"PA": [], "RA": [], "NA": []}
    knn = {"PA": [], "RA": [], "NA": []}
    for s in samples:
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prepared(s)
        with open(s.semantic, encoding="utf-8") as f:
            clef = f.read().split()[0]
        for tag, clf in (("geo", None), ("knn", dur_clf)):
            pred = decode(mask, nostaff, metrics, clef, dur_clf=clf)
            sc = note_scores(pred, gt)
            d = geo if tag == "geo" else knn
            d["PA"].append(sc["pitch_accuracy"])
            d["RA"].append(sc["rhythm_accuracy"])
            d["NA"].append(sc["note_accuracy"])
    return geo, knn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.7 * len(samples))

    clf = DurationClassifier(k=args.k).fit_samples(samples[:split], prepared)
    print(f"train notes={len(clf.knn.y)}")

    geo, knn = evaluate(samples[split:], clf)
    print(f"test={len(geo['NA'])}")
    for m in ("PA", "RA", "NA"):
        print(f"  {m}: géo {np.mean(geo[m]):.1f}  ->  kNN {np.mean(knn[m]):.1f}")


if __name__ == "__main__":
    main()
