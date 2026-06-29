from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

from .io_primus import PrimusSample, load_gray, list_samples, parse_semantic
from .preprocess import preprocess


def _save_mask(mask: np.ndarray, path: str) -> None:
    Image.fromarray(np.where(mask, 0, 255).astype(np.uint8)).save(path)


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
        out = os.path.join(out_dir, f"{sample.sample_id}_{tag}_bin.png")
        _save_mask(mask, out)
        print(f"  -> {out}  ({int(mask.sum())} px encre)")


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
