"""Harnais de vérification WP3 : compare le décodage Python et C++ sur les
mêmes incipits de sample_corpus/, et mesure le temps d'exécution des deux.

Le binaire C++ est compilé sous WSL (pas de CMake natif Windows disponible) ;
ce script tourne côté Windows (accès à uv/omr/numpy/Pillow) et invoque
l'exécutable en mode --bench (une seule invocation pour tout le corpus, modèle
chargé une fois, pour une comparaison de temps équitable — sans quoi le coût
de démarrage de WSL/du processus dominerait le temps mesuré).

Usage (depuis la racine du dépôt, Windows) :
    uv run python cpp/verify/compare.py
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

from omr.decode import decode
from omr.duration_knn import DurationClassifier
from omr.io_primus import list_samples, load_gray
from omr.preprocess import preprocess
from omr.staff import estimate_metrics, remove_staff_lines_combined, staff_line_rows


def to_wsl_path(win_path: str) -> str:
    win_path = os.path.abspath(win_path).replace("\\", "/")
    drive, rest = win_path.split(":", 1)
    return f"/mnt/{drive.lower()}{rest}"


def decode_python(sample, dur_clf):
    gray = load_gray(sample.png)
    mask = preprocess(gray)
    metrics = estimate_metrics(mask)
    rows = staff_line_rows(mask)
    nostaff = remove_staff_lines_combined(mask, metrics, rows)
    with open(sample.semantic, encoding="utf-8") as f:
        clef = f.read().split()[0]
    notes = decode(mask, nostaff, metrics, clef, dur_clf=dur_clf)
    return [(n["pitch"], n["duration"]) for n in notes]


def parse_note_seq(text: str):
    notes = []
    for tok in text.split():
        pitch, _, duration = tok.rpartition("/")
        notes.append((pitch, duration))
    return notes


def run_cpp_bench(exe_wsl, corpus_wsl, model_wsl, n):
    cmd = ["wsl", exe_wsl, "--bench", "--corpus", corpus_wsl, "--model", model_wsl]
    if n is not None:
        cmd += ["--n", str(n)]
    t0 = time.perf_counter()
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - t0

    per_sample = {}
    for line in out.stdout.splitlines():
        sample_id, _, seq = line.partition("\t")
        per_sample[sample_id] = parse_note_seq(seq)

    avg_ms = None
    m = re.search(r"([\d.]+)\s*ms/echantillon", out.stderr)
    if m:
        avg_ms = float(m.group(1))
    return per_sample, wall, avg_ms


def main() -> None:
    ap = argparse.ArgumentParser(description="Comparaison Python/C++ (WP3, issue #10)")
    ap.add_argument("--corpus", default="sample_corpus")
    ap.add_argument("--exe", default="cpp/build/omr_cpp")
    ap.add_argument("--model", default="models/duration_knn.npz")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    exe_wsl = to_wsl_path(args.exe)
    corpus_wsl = to_wsl_path(args.corpus)
    model_wsl = to_wsl_path(args.model)

    samples = [s for s in list_samples(args.corpus, limit=args.n) if os.path.exists(s.png)]
    dur_clf = DurationClassifier.load(os.path.abspath(args.model))

    py_times = []
    py_notes_by_id = {}
    for sample in samples:
        t0 = time.perf_counter()
        py_notes_by_id[sample.sample_id] = decode_python(sample, dur_clf)
        py_times.append(time.perf_counter() - t0)

    cpp_notes_by_id, cpp_wall, cpp_avg_ms = run_cpp_bench(exe_wsl, corpus_wsl, model_wsl, args.n)

    n_total = len(samples)
    n_match = 0
    mismatches = []
    n_notes = 0
    n_pitch_diff = 0
    n_dur_diff = 0
    n_len_diff = 0
    for sample in samples:
        py_notes = py_notes_by_id[sample.sample_id]
        cpp_notes = cpp_notes_by_id.get(sample.sample_id, [])
        if py_notes == cpp_notes:
            n_match += 1
        else:
            mismatches.append((sample.sample_id, py_notes, cpp_notes))
        if len(py_notes) != len(cpp_notes):
            n_len_diff += 1
            continue
        for (pp, pd), (cp, cd) in zip(py_notes, cpp_notes):
            n_notes += 1
            n_pitch_diff += pp != cp
            n_dur_diff += pd != cd

    py_avg_ms = 1000 * sum(py_times) / len(py_times)
    lines = [
        f"echantillons compares : {n_total}",
        f"correspondance exacte : {n_match}/{n_total} ({100.0 * n_match / n_total:.1f} %)",
        f"temps moyen python (decode seul)     : {py_avg_ms:.3f} ms/echantillon",
        f"temps moyen c++ (decode seul)         : {cpp_avg_ms:.3f} ms/echantillon"
        if cpp_avg_ms is not None else "temps moyen c++ : indisponible",
        f"acceleration (decode seul)           : x{py_avg_ms / cpp_avg_ms:.1f}"
        if cpp_avg_ms else "",
        f"temps total c++ (process + {n_total} echantillons) : {1000*cpp_wall:.1f} ms",
        f"echantillons de longueur differente  : {n_len_diff}/{n_total}",
        f"notes comparees (longueur egale)     : {n_notes}",
        f"  dont hauteur differente            : {n_pitch_diff}",
        f"  dont duree differente              : {n_dur_diff}",
    ]
    print("\n".join(lines))

    if mismatches:
        print(f"\nDivergences ({len(mismatches)}/{n_total}) :")
        for sample_id, py_notes, cpp_notes in mismatches:
            print(f"  {sample_id}")
            print(f"    python: {py_notes}")
            print(f"    c++   : {cpp_notes}")


if __name__ == "__main__":
    main()
