# Livrable 2 — Méthodes d'évaluation (OMR)

Choix et implémentation des métriques d'évaluation pour l'analyse automatique de partitions.

## Contenu

- `main.tex` — description des critères (compilable avec `pdflatex` + `biber`).
- `base.tex` — description de la base d'évaluation (datasets, liens, autorisations).
- `references.bib` — bibliographie partagée par les deux `.tex`.
- `tools/metrics.py` — outils de calcul des scores (NumPy seul).
- `tools/evaluate.py` — lit une vérité-terrain et une prédiction (JSON) et affiche les scores.
- `data/README.md` — format JSON d'entrée (note interne).
- `tests/` — tests unitaires (`unittest`).

## Les trois critères

| # | Métrique | Outil |
|---|---|---|
| 1 | Pitch Accuracy (PA) — % hauteurs correctes | `note_scores` |
| 2 | Rhythm Accuracy (RA) + Note Accuracy (NA) — % durées / notes complètes correctes | `note_scores` |
| 3 | F1 de détection des têtes de notes (IoU ≥ 0.5) | `detection_f1` |

PA et RA sont calculées après alignement des séquences par distance d'édition.

## Utilisation

```
python tools/evaluate.py --gt vérité.json --pred prédiction.json
python -m unittest discover -s tests
```

Dépendance : `numpy`.
