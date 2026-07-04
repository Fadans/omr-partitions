# Reconnaissance — Analyse automatique de partitions musicales (OMR)

SAE TIFO 1 / TIFO 2 / MLRF — Thibault Bourdoiseau & Léo Li.

Prototype Python (WP2) qui transforme l'image d'une partition monophonique
(base CameraPrIMuS) en une séquence de notes (hauteur + durée), évaluée par les
métriques du WP1. NumPy + Pillow uniquement, pour faciliter le portage C++ (WP3).

## Structure

| Chemin | Contenu |
|---|---|
| `livrable1_etat_de_lart/` | État de l'art (WP1-T1) |
| `livrable2_evaluation/` | Métriques et outils d'évaluation (WP1-T2) |
| `omr/` | Prototype Python (WP2) |
| `tests/` | Tests unitaires |
| `base_1/`, `base_2/` | Données (non versionnées) |

## Pipeline (`omr/`)

1. `preprocess.py` — niveaux de gris → médian → binarisation (Otsu / adaptatif). ✅
2. `io_primus.py` — chargement image + parsing de la vérité-terrain `.semantic`. ✅
3. `staff.py` — détection et suppression des portées (ouverture morphologique). ✅
4. `segment.py` — composantes connexes → symboles. ✅
5. `morph.py`, `heads.py` — morphologie + détection des têtes (granulométrie). ✅
6. `descriptors.py`, `knn.py` — descripteurs de forme (Hu, projections) + k-NN. ✅
7. `decode.py` — position → hauteur (clef), forme → durée → séquence. ✅
8. `midi.py` — export Standard MIDI File (pur Python). ✅

Résultats finaux (test disjoint, n=800) : Note Accuracy **74 %** (sous-ensemble
simple) / **56 %** (général), PA diatonique 84 %, RA 71 % ; clef reconnue à
99 % (propre) / 92 % (distordu). Voir `BILAN_WP2.md` (bilan V1/V2, décisions,
limites), `SUIVI.md` (journal des essais) et `CORRESPONDANCE_COURS.md`
(notions de cours mobilisées).

## Utilisation

```sh
uv sync --extra dev
uv run pytest -q
uv run python -m omr.cli --id 000104970-1_1_1   # démo prétraitement
```
