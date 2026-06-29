# Outils d'évaluation — format d'entrée

## `detection_f1(pred_boxes, gt_boxes, iou_threshold=0.5, scores=None)`

F1 de détection des têtes de notes.

- `pred_boxes`, `gt_boxes` : listes de boîtes `[x1, y1, x2, y2]` en pixels (coin haut-gauche, coin
  bas-droite).
- `scores` (optionnel) : liste de confiances des prédictions, même longueur que `pred_boxes` ; ordonne
  l'appariement glouton.
- Retour : `{"precision", "recall", "f1", "tp", "fp", "fn"}`.

## `note_scores(pred_notes, gt_notes)`

Pitch Accuracy, Rhythm Accuracy et Note Accuracy (en %), après alignement par distance d'édition.

- `pred_notes`, `gt_notes` : listes de notes dans l'ordre de lecture (gauche → droite). Chaque note est un
  dict avec au moins :
  - `"pitch"` : nom + octave, ex. `"C4"`.
  - `"duration"` : code de durée, ex. `"q"` (`w` ronde, `h` blanche, `q` noire, `e` croche, `s` double
    croche…).
- Retour : `{"pitch_accuracy", "rhythm_accuracy", "note_accuracy"}`.

## `evaluate.py` — format fichier JSON

`evaluate.py` lit une vérité-terrain et une prédiction au même format, puis affiche les scores. Une image =
un fichier JSON.

```json
{
  "notes": [
    {"pitch": "C4", "duration": "q", "bbox": [x1, y1, x2, y2], "score": 0.9},
    {"pitch": "E4", "duration": "h", "bbox": [x1, y1, x2, y2]}
  ]
}
```

`bbox` alimente `detection_f1`, `pitch`/`duration` alimentent `note_scores`, `score` est optionnel.
