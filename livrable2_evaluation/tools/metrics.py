from __future__ import annotations

import numpy as np


def iou_box(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def detection_f1(pred_boxes, gt_boxes, iou_threshold: float = 0.5, scores=None) -> dict:
    pred_boxes, gt_boxes = list(pred_boxes), list(gt_boxes)
    order = range(len(pred_boxes))
    if scores is not None:
        order = sorted(order, key=lambda i: scores[i], reverse=True)

    used, tp = set(), 0
    for i in order:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou_box(pred_boxes[i], g)
            if v > best_iou:
                best_iou, best_j = v, j
        if best_j >= 0 and best_iou >= iou_threshold:
            used.add(best_j)
            tp += 1

    fp, fn = len(pred_boxes) - tp, len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def _align(a, b):
    n, m = len(a), len(b)
    d = np.zeros((n + 1, m + 1), dtype=int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)

    pairs, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (0 if a[i - 1] == b[j - 1] else 1):
            pairs.append((i - 1, j - 1)); i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            pairs.append((i - 1, None)); i -= 1
        else:
            pairs.append((None, j - 1)); j -= 1
    pairs.reverse()
    return pairs


def note_scores(pred_notes, gt_notes) -> dict:
    pred = [(n["pitch"], n["duration"]) for n in pred_notes]
    gt = [(n["pitch"], n["duration"]) for n in gt_notes]
    n_ref = len(gt)

    pitch_ok = dur_ok = both_ok = 0
    for ia, ib in _align(pred, gt):
        if ia is None or ib is None:
            continue
        p_pitch, p_dur = pred[ia]
        g_pitch, g_dur = gt[ib]
        ok_pitch = int(p_pitch == g_pitch)
        ok_dur = int(p_dur == g_dur)
        pitch_ok += ok_pitch
        dur_ok += ok_dur
        both_ok += ok_pitch & ok_dur

    if n_ref == 0:
        return {"pitch_accuracy": 100.0, "rhythm_accuracy": 100.0, "note_accuracy": 100.0}
    return {
        "pitch_accuracy": 100.0 * pitch_ok / n_ref,
        "rhythm_accuracy": 100.0 * dur_ok / n_ref,
        "note_accuracy": 100.0 * both_ok / n_ref,
    }
