from __future__ import annotations

import argparse
import os

from omr.dataset import prepared
from omr.duration_knn import DurationClassifier
from omr.io_primus import list_samples


def main():
    ap = argparse.ArgumentParser(description="Entraîne et sauve le classifieur de durées k-NN")
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="models/duration_knn.npz")
    args = ap.parse_args()

    samples = list_samples(args.corpus, limit=args.n)
    clf = DurationClassifier(k=args.k).fit_samples(samples, prepared)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    clf.save(args.out)
    print(f"modèle entraîné sur {len(clf.knn.y)} notes -> {args.out}")


if __name__ == "__main__":
    main()
