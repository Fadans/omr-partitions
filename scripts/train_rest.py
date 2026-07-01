from __future__ import annotations

import argparse
import os

from omr.dataset import prepared
from omr.io_primus import list_samples
from omr.rest_knn import RestClassifier


def main():
    ap = argparse.ArgumentParser(description="Entraîne et sauve le classifieur de silences k-NN")
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="models/rest_knn.npz")
    args = ap.parse_args()

    samples = list_samples(args.corpus, limit=args.n)
    clf = RestClassifier(k=args.k).fit_samples(samples, prepared)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    clf.save(args.out)
    n_pos = sum(1 for lab in clf.knn.y if lab != "none")
    print(f"modèle silences : {len(clf.knn.y)} candidats ({n_pos} silences) -> {args.out}")


if __name__ == "__main__":
    main()
