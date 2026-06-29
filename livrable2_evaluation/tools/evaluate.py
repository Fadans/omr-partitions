from __future__ import annotations

import argparse
import json

from metrics import detection_f1, note_scores


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate(gt, pred):
    gt_notes, pred_notes = gt["notes"], pred["notes"]
    det = detection_f1(
        [n["bbox"] for n in pred_notes],
        [n["bbox"] for n in gt_notes],
        scores=[n.get("score", 1.0) for n in pred_notes],
    )
    notes = note_scores(pred_notes, gt_notes)
    return {"detection": det, "notes": notes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()
    r = evaluate(_load(args.gt), _load(args.pred))
    d, n = r["detection"], r["notes"]
    print(f"Detection  : precision={d['precision']:.3f} recall={d['recall']:.3f} "
          f"F1={d['f1']:.3f} (TP={d['tp']} FP={d['fp']} FN={d['fn']})")
    print(f"Pitch      : {n['pitch_accuracy']:.1f} %")
    print(f"Rhythm     : {n['rhythm_accuracy']:.1f} %")
    print(f"Note (both): {n['note_accuracy']:.1f} %")


if __name__ == "__main__":
    main()
