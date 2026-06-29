from __future__ import annotations

import argparse

from omr.dataset import keysig_strip_descriptor, key_token
from omr.io_primus import list_samples
from omr.key_signature import key_to_accidentals
from omr.knn import NearestNeighbor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="base_1/Corpus")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--bins", type=int, default=24)
    ap.add_argument("--distorted", action="store_true")
    args = ap.parse_args()

    samples = list_samples(args.corpus, limit=args.n)
    split = int(0.7 * len(samples))

    X, y = [], []
    for s in samples[:split]:
        d = keysig_strip_descriptor(s, args.distorted, args.bins)
        if d is not None:
            X.append(d)
            y.append(key_token(s).split("-")[-1])
    clf = NearestNeighbor(k=args.k).fit(X, y)
    print(f"train={len(y)} classes={len(set(y))}")

    ok_key = ok_count = ok_kind = total = 0
    for s in samples[split:]:
        d = keysig_strip_descriptor(s, args.distorted, args.bins)
        if d is None:
            continue
        pred = clf.predict(d)[0]
        gt = key_token(s).split("-")[-1]
        pk, pc = key_to_accidentals(pred)
        gk, gc = key_to_accidentals(gt)
        total += 1
        ok_key += pred == gt
        ok_count += pc == gc
        ok_kind += (pk == gk) or (gc == 0 and pc == 0)

    print(f"test={total}")
    print(f"  tonalité exacte             : {100 * ok_key / total:.1f} %")
    print(f"  armure (type+nombre) exacte : {100 * (ok_key) / total:.1f} %")
    print(f"  nombre d'altérations exact  : {100 * ok_count / total:.1f} %")
    print(f"  type exact                  : {100 * ok_kind / total:.1f} %")


if __name__ == "__main__":
    main()
