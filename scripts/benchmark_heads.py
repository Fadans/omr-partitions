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
from omr.heads import detect_heads, detect_heads_template  # noqa: E402
from omr.io_primus import list_samples, parse_semantic  # noqa: E402


def evaluate(samples, dur_clf, detector):
    acc = {"pa": [], "pa_dia": [], "ra": [], "na": []}
    ratio = []
    for s in samples:
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prepared(s)
        clef = open(s.semantic, encoding="utf-8").read().split()[0]
        heads = detector(nostaff, metrics)
        pred = decode(mask, nostaff, metrics, clef, heads=heads, dur_clf=dur_clf)
        full = note_scores(pred, gt)
        dia = note_scores(
            [{"pitch": n["pitch"].replace("#", "").replace("b", ""), "duration": n["duration"]} for n in pred],
            [{"pitch": n["pitch"].replace("#", "").replace("b", ""), "duration": n["duration"]} for n in gt])
        acc["pa"].append(full["pitch_accuracy"]); acc["pa_dia"].append(dia["pitch_accuracy"])
        acc["ra"].append(full["rhythm_accuracy"]); acc["na"].append(full["note_accuracy"])
        ratio.append(len(pred) / len(gt))
    return acc, np.mean(ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.7 * len(samples))
    dur = DurationClassifier(k=args.k).fit_samples(samples[:split], prepared)
    test = samples[split:]

    for name, det in (("morphologie (actuel)", detect_heads),
                      ("gabarit (template)", detect_heads_template)):
        acc, ratio = evaluate(test, dur, det)
        m = lambda k: np.mean(acc[k]) if acc[k] else 0.0
        print(f"{name:22} | ratio têtes/GT {ratio:.3f} | "
              f"PA {m('pa'):.1f} | PA_dia {m('pa_dia'):.1f} | "
              f"RA {m('ra'):.1f} | NA {m('na'):.1f}")


if __name__ == "__main__":
    main()
