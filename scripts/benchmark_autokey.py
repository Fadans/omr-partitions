from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("livrable2_evaluation", "tools"))
from metrics import note_scores  # noqa: E402

from omr.dataset import keysig_strip_descriptor, key_token, prepared  # noqa: E402
from omr.decode import decode  # noqa: E402
from omr.duration_knn import DurationClassifier  # noqa: E402
from omr.io_primus import list_samples, parse_semantic  # noqa: E402
from omr.key_signature import alteration_map, key_to_accidentals  # noqa: E402
from omr.knn import NearestNeighbor  # noqa: E402


def strip_acc(notes):
    return [{"pitch": n["pitch"].replace("#", "").replace("b", ""),
             "duration": n["duration"]} for n in notes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.6 * len(samples))

    X, y = [], []
    for s in samples[:split]:
        d = keysig_strip_descriptor(s)
        if d is not None:
            X.append(d)
            y.append(key_token(s).split("-")[-1])
    clf = NearestNeighbor(k=args.k).fit(X, y)
    dur_clf = DurationClassifier(k=args.k).fit_samples(samples[:split], prepared)

    modes = {"none": [], "auto": [], "oracle": []}
    for s in samples[split:]:
        toks_clef = None
        with open(s.semantic, encoding="utf-8") as f:
            toks = f.read().split()
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prepared(s)
        clef = toks[0]

        d = keysig_strip_descriptor(s)
        pred_key = clf.predict(d)[0] if d is not None else "CM"
        alts = {
            "none": None,
            "auto": alteration_map(*key_to_accidentals(pred_key)),
            "oracle": alteration_map(*key_to_accidentals(key_token(s))),
        }
        for mode, alt in alts.items():
            pred = decode(mask, nostaff, metrics, clef, alt_map=alt, dur_clf=dur_clf)
            modes[mode].append(note_scores(pred, gt)["note_accuracy"])

    print(f"test={len(modes['none'])}")
    for mode in ("none", "auto", "oracle"):
        print(f"  NA {mode:7}: {np.mean(modes[mode]):.1f} %")


if __name__ == "__main__":
    main()
