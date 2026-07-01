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
from omr.key_signature import alteration_map, key_to_accidentals  # noqa: E402

HOLLOW = {"whole", "half"}


def tokens(sample):
    with open(sample.semantic, encoding="utf-8") as f:
        return f.read().split()


def clef_token(toks):
    return toks[0]


def key_token(toks):
    return toks[1] if len(toks) > 1 and toks[1].startswith("keySignature") else "CM"


def is_simple(toks) -> bool:
    if key_to_accidentals(key_token(toks))[1] != 0:
        return False
    for t in toks:
        if t.startswith("note") or t.startswith("rest"):
            base = t.split("-")[-1].split("_")[0].rstrip(".")
            if base in HOLLOW:
                return False
    return True


def strip_acc(notes):
    return [{"pitch": n["pitch"].replace("#", "").replace("b", ""),
             "duration": n["duration"]} for n in notes]


def evaluate(samples, oracle, dur_clf=None):
    acc = {"pa": [], "pa_dia": [], "ra": [], "na": []}
    for s in samples:
        toks = tokens(s)
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prepared(s)
        alt = alteration_map(*key_to_accidentals(key_token(toks))) if oracle else None
        pred = decode(mask, nostaff, metrics, clef_token(toks), alt_map=alt,
                      dur_clf=dur_clf)
        full = note_scores(pred, gt)
        dia = note_scores(strip_acc(pred), strip_acc(gt))
        acc["pa"].append(full["pitch_accuracy"])
        acc["pa_dia"].append(dia["pitch_accuracy"])
        acc["ra"].append(full["rhythm_accuracy"])
        acc["na"].append(full["note_accuracy"])
    return acc


def row(name, n, acc):
    m = lambda k: np.mean(acc[k]) if acc[k] else 0.0
    print(f"| {name:24} | {n:4} | {m('pa'):5.1f} | {m('pa_dia'):5.1f} "
          f"| {m('ra'):5.1f} | {m('na'):5.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--key-oracle", action="store_true")
    ap.add_argument("--geom", action="store_true",
                    help="durées géométriques au lieu du classifieur k-NN")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    samples = list_samples(args.corpus, limit=args.n)
    # split train/test : le classifieur de durées est entraîné sur la partie
    # train, l'évaluation ne porte que sur la partie test (disjointe).
    split = int(0.7 * len(samples))
    train, test = samples[:split], samples[split:]

    dur_clf = None
    if not args.geom:
        dur_clf = DurationClassifier(k=args.k).fit_samples(train, prepared)

    simple = [s for s in test if is_simple(tokens(s))]

    kind = "durées géométriques" if args.geom else "durées k-NN"
    title = " (armure oracle)" if args.key_oracle else ""
    print(f"Banc d'essai PrIMuS{title} — {kind} — test={len(test)} incipits, "
          f"{len(simple)} simples\n")
    print("| Sous-ensemble            |    N | PA st | PA di |    RA |    NA |")
    print("|--------------------------|------|-------|-------|-------|-------|")
    row("PrIMuS simple", len(simple), evaluate(simple, args.key_oracle, dur_clf))
    row("PrIMuS général", len(test), evaluate(test, args.key_oracle, dur_clf))
    print("\nPA st = pitch strict (avec altérations) ; PA di = diatonique "
          "(sans) ; RA = rythme ; NA = note complète. Moyennes en %.")


if __name__ == "__main__":
    main()
