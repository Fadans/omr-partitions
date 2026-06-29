# Format d'entrée des outils (note interne)

Document interne — la description officielle de la base est dans `base.tex`.

`tools/evaluate.py` lit une vérité-terrain et une prédiction au même format JSON. On convertit l'annotation
de chaque base (DeepScoresV2, Camera-PrIMuS) vers ce format, puis on calcule les scores.

## Format JSON

Une image = un fichier. Vérité-terrain et prédiction partagent le même schéma.

```json
{
  "notes": [
    {"pitch": "C4", "duration": "q", "bbox": [x1, y1, x2, y2], "score": 0.9},
    {"pitch": "E4", "duration": "h", "bbox": [x1, y1, x2, y2]}
  ]
}
```

- `notes` : notes dans l'ordre de lecture (gauche → droite).
- `pitch` : nom + octave (ex. `C4`). Utilisé par la Pitch Accuracy.
- `duration` : code de durée (`w` ronde, `h` blanche, `q` noire, `e` croche, `s` double croche…). Utilisé par
  la Rhythm Accuracy.
- `bbox` : `[x1, y1, x2, y2]` en pixels (coin haut-gauche puis bas-droite). Utilisé par le F1 de détection.
- `score` : confiance de la prédiction (facultatif) ; ordonne l'appariement de détection.
