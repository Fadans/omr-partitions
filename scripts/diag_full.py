from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.join("livrable2_evaluation", "tools"))
from metrics import _align, note_scores  # noqa: E402

from omr.dataset import prepared  # noqa: E402
from omr.decode import decode  # noqa: E402
from omr.io_primus import list_samples, parse_semantic  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    args = ap.parse_args()

    samples = list_samples("base_1/Corpus", limit=args.n)

    pa = ra = na = 0.0
    n_used = 0
    dur_conf = defaultdict(Counter)      # gt_dur -> Counter(pred_dur) on matched pairs
    dur_conf_pok = defaultdict(Counter)  # same but only pitch-correct pairs
    gt_dur_hist = Counter()
    n_gt = n_pred = 0
    matched = pitch_ok_c = dur_ok_c = unmatched_gt = unmatched_pred = 0

    for s in samples:
        gt = [n for n in parse_semantic(s.semantic) if n["pitch"] != "rest"]
        if not gt:
            continue
        mask, nostaff, metrics, _ = prepared(s)
        with open(s.semantic, encoding="utf-8") as f:
            clef = f.read().split()[0]
        pred = decode(mask, nostaff, metrics, clef)

        sc = note_scores(pred, gt)
        pa += sc["pitch_accuracy"]; ra += sc["rhythm_accuracy"]; na += sc["note_accuracy"]
        n_used += 1
        n_gt += len(gt); n_pred += len(pred)
        for g in gt:
            gt_dur_hist[g["duration"]] += 1

        pt = [(n["pitch"], n["duration"]) for n in pred]
        gtt = [(n["pitch"], n["duration"]) for n in gt]
        for ia, ib in _align(pt, gtt):
            if ia is None:
                unmatched_pred += 1; continue
            if ib is None:
                unmatched_gt += 1; continue
            matched += 1
            pp, pd = pt[ia]; gp, gd = gtt[ib]
            ok_p = pp == gp
            pitch_ok_c += ok_p
            dur_ok_c += pd == gd
            dur_conf[gd][pd] += 1
            if ok_p:
                dur_conf_pok[gd][pd] += 1

    print(f"n={n_used}  GT notes={n_gt}  pred notes={n_pred}  (ratio {n_pred/n_gt:.3f})")
    print(f"PA {pa/n_used:.1f}  RA {ra/n_used:.1f}  NA {na/n_used:.1f}")
    print(f"matched={matched} pitch_ok={pitch_ok_c} dur_ok={dur_ok_c} "
          f"unmatched_gt={unmatched_gt} unmatched_pred={unmatched_pred}")
    print(f"  pitch acc on matched: {100*pitch_ok_c/matched:.1f}  "
          f"dur acc on matched: {100*dur_ok_c/matched:.1f}")

    durs = ["whole", "half", "quarter", "eighth", "sixteenth", "thirty_second"]
    print("\nGT duration distribution:")
    tot = sum(gt_dur_hist.values())
    for d in durs:
        if gt_dur_hist.get(d):
            print(f"  {d:14} {gt_dur_hist[d]:5}  {100*gt_dur_hist[d]/tot:4.1f}%")
    others = tot - sum(gt_dur_hist.get(d, 0) for d in durs)
    if others:
        print(f"  (other/dotted)  {others:5}  {100*others/tot:4.1f}%")

    def show(conf, title):
        print(f"\n{title} (row=GT, col=pred):")
        cols = durs
        print("  " + "".join(f"{c[:5]:>7}" for c in cols) + "   acc")
        for g in durs:
            row = conf.get(g)
            if not row:
                continue
            rt = sum(row.values())
            cells = "".join(f"{row.get(c,0):>7}" for c in cols)
            print(f"  {g[:2]}{cells}  {100*row.get(g,0)/rt:5.1f}")

    show(dur_conf, "Duration confusion (all matched)")
    show(dur_conf_pok, "Duration confusion (pitch-correct only)")


if __name__ == "__main__":
    main()
