from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("livrable2_evaluation", "tools"))
from metrics import note_scores  # noqa: E402

from omr.decode import decode  # noqa: E402
from omr.duration_knn import DurationClassifier  # noqa: E402
from omr.heads import detect_heads  # noqa: E402
from omr.io_primus import list_samples, load_gray, parse_semantic  # noqa: E402
from omr.preprocess import preprocess  # noqa: E402
from omr.segment import segment  # noqa: E402
from omr.staff import (estimate_metrics, remove_staff_lines_combined,  # noqa: E402
                       remove_staff_lines_stable_paths)

REMOVERS = {
    "combined": remove_staff_lines_combined,
    "stable_paths": remove_staff_lines_stable_paths,
}


def make_prepared(remover):
    def prep(sample, distorted=False):
        path = sample.distorted if distorted else sample.png
        mask = preprocess(load_gray(path), adaptive=distorted)
        metrics = estimate_metrics(mask)
        nostaff = remover(mask, metrics)
        comps = segment(nostaff, min_area=metrics.line_height * metrics.line_space)
        return mask, nostaff, metrics, comps
    return prep


def evaluate(samples, prep, dur):
    acc = {"pa": [], "pa_dia": [], "ra": [], "na": []}
    ratio = []
    for s in samples:
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prep(s)
        clef = open(s.semantic, encoding="utf-8").read().split()[0]
        pred = decode(mask, nostaff, metrics, clef, dur_clf=dur)
        full = note_scores(pred, gt)
        strip = lambda ns: [{"pitch": n["pitch"].replace("#", "").replace("b", ""),
                             "duration": n["duration"]} for n in ns]
        dia = note_scores(strip(pred), strip(gt))
        acc["pa"].append(full["pitch_accuracy"]); acc["pa_dia"].append(dia["pitch_accuracy"])
        acc["ra"].append(full["rhythm_accuracy"]); acc["na"].append(full["note_accuracy"])
        ratio.append(len(pred) / len(gt))
    return acc, float(np.mean(ratio))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    samples = list_samples("base_1/Corpus", limit=args.n)
    split = int(0.7 * len(samples))

    for name, remover in REMOVERS.items():
        prep = make_prepared(remover)
        dur = DurationClassifier(k=args.k).fit_samples(samples[:split], prep)
        acc, ratio = evaluate(samples[split:], prep, dur)
        m = lambda k: np.mean(acc[k]) if acc[k] else 0.0
        print(f"{name:14} | ratio têtes/GT {ratio:.3f} | PA {m('pa'):.1f} | "
              f"PA_dia {m('pa_dia'):.1f} | RA {m('ra'):.1f} | NA {m('na'):.1f}")


if __name__ == "__main__":
    main()
