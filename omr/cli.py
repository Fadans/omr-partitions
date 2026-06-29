from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from .io_primus import PrimusSample, load_gray, list_samples, parse_semantic
from .preprocess import preprocess
from .segment import Component, segment
from .staff import estimate_metrics, remove_staff_lines, staff_line_rows


def _save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


def _save_boxes(mask: np.ndarray, comps: list[Component], path: str) -> None:
    rgb = np.where(mask, 0, 255).astype(np.uint8)
    rgb = np.stack([rgb] * 3, axis=-1)
    for c in comps:
        rgb[c.y1:c.y2, c.x1, :] = (255, 0, 0)
        rgb[c.y1:c.y2, c.x2 - 1, :] = (255, 0, 0)
        rgb[c.y1, c.x1:c.x2, :] = (255, 0, 0)
        rgb[c.y2 - 1, c.x1:c.x2, :] = (255, 0, 0)
    Image.fromarray(rgb).save(path)


def demo(sample: PrimusSample, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    notes = parse_semantic(sample.semantic)
    print(f"{sample.sample_id}: {len(notes)} notes/silences")
    for n in notes[:12]:
        print(f"  {n['pitch']:>5}  {n['duration']}")

    for tag, path, adaptive in (
        ("clean", sample.png, False),
        ("distorted", sample.distorted, True),
    ):
        if not os.path.exists(path):
            continue
        mask = preprocess(load_gray(path), adaptive=adaptive)
        _save_mask(mask, os.path.join(out_dir, f"{sample.sample_id}_{tag}_bin.png"))

        metrics = estimate_metrics(mask)
        rows = staff_line_rows(mask)
        nostaff = remove_staff_lines(mask, metrics)
        _save_mask(nostaff, os.path.join(out_dir, f"{sample.sample_id}_{tag}_nostaff.png"))

        comps = segment(nostaff, min_area=metrics.line_height * metrics.line_space)
        _save_boxes(nostaff, comps, os.path.join(out_dir, f"{sample.sample_id}_{tag}_boxes.png"))
        print(f"  [{tag}] interligne={metrics.line_space} épaisseur={metrics.line_height} "
              f"lignes={len(rows)} encre {int(mask.sum())}->{int(nostaff.sum())} "
              f"symboles={len(comps)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Demo prétraitement OMR (PrIMuS)")
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--id", default=None, help="id d'échantillon précis")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    if args.id:
        sample = PrimusSample(args.id, os.path.join(args.corpus, args.id))
    else:
        sample = list_samples(args.corpus, limit=1)[0]
    demo(sample, args.out)


if __name__ == "__main__":
    main()
