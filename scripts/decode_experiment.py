from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("livrable2_evaluation", "tools"))
from metrics import note_scores  # noqa: E402

from omr.dataset import prepared  # noqa: E402
from omr.decode import decode  # noqa: E402
from omr.io_primus import list_samples, parse_semantic  # noqa: E402
from omr.key_signature import alteration_map, key_to_accidentals  # noqa: E402


def _strip(notes):
    return [{"pitch": n["pitch"].replace("#", "").replace("b", ""),
             "duration": n["duration"]} for n in notes]


def clef_token(sample):
    with open(sample.semantic, encoding="utf-8") as f:
        return f.read().split()[0]


def key_token(sample):
    with open(sample.semantic, encoding="utf-8") as f:
        toks = f.read().split()
    return toks[1] if len(toks) > 1 and toks[1].startswith("keySignature") else "CM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--distorted", action="store_true")
    ap.add_argument("--key-oracle", action="store_true",
                    help="applique l'armure de la vérité-terrain (plafond)")
    args = ap.parse_args()

    agg = {"pa": [], "ra": [], "na": [], "pa_dia": []}
    n_done = 0
    for s in list_samples(args.corpus, limit=args.n):
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prepared(s, distorted=args.distorted)
        alt = alteration_map(*key_to_accidentals(key_token(s))) if args.key_oracle else None
        pred = decode(mask, nostaff, metrics, clef_token(s), alt_map=alt)

        full = note_scores(pred, gt)
        dia = note_scores(_strip(pred), _strip(gt))
        agg["pa"].append(full["pitch_accuracy"])
        agg["ra"].append(full["rhythm_accuracy"])
        agg["na"].append(full["note_accuracy"])
        agg["pa_dia"].append(dia["pitch_accuracy"])
        n_done += 1

    kind = "distordu" if args.distorted else "propre"
    print(f"[{kind}] {n_done} partitions")
    print(f"  Pitch Accuracy (strict, avec altération) : {np.mean(agg['pa']):.1f} %")
    print(f"  Pitch Accuracy (diatonique, sans altér.) : {np.mean(agg['pa_dia']):.1f} %")
    print(f"  Rhythm Accuracy                          : {np.mean(agg['ra']):.1f} %")
    print(f"  Note Accuracy (hauteur+durée)            : {np.mean(agg['na']):.1f} %")


if __name__ == "__main__":
    main()
