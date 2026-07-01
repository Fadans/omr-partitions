from __future__ import annotations

import argparse

import numpy as np

from omr.io_primus import list_samples, load_gray
from omr.preprocess import preprocess
from omr.staff import staff_line_rows, staff_lines_hough


def multiple_of_five(n):
    return n >= 5 and n % 5 == 0


def run(samples, distorted):
    proj_ok = hough_ok = 0
    angles = []
    n = 0
    for s in samples:
        path = s.distorted if distorted else s.png
        try:
            mask = preprocess(load_gray(path), adaptive=distorted)
        except Exception:
            continue
        n += 1
        proj = staff_line_rows(mask)
        deg, hough = staff_lines_hough(mask)
        proj_ok += multiple_of_five(len(proj))
        hough_ok += multiple_of_five(len(hough))
        angles.append(abs(deg))
    return n, proj_ok, hough_ok, float(np.mean(angles)) if angles else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    samples = list_samples("base_1/Corpus", limit=args.n)
    for distorted in (False, True):
        n, p, hgh, ang = run(samples, distorted)
        tag = "DÉFORMÉES" if distorted else "propres"
        print(f"[{tag}] n={n} | projection 5-lignes OK: {100*p/n:.1f}% | "
              f"Hough 5-lignes OK: {100*hgh/n:.1f}% | inclinaison moy. |deg|={ang:.2f}")


if __name__ == "__main__":
    main()
