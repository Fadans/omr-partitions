# Prototype OMR — utilisation

Reconnaissance de partitions monophoniques : image → notes (hauteur + durée) → MIDI.
Notions vues en cours, sans réseau de neurones. NumPy et Pillow uniquement.

## Installation

```sh
uv sync --extra dev
```

## Tests unitaires

```sh
uv run pytest -q
```

## Test rapide (échantillon fourni)

Le dossier `sample_corpus/` contient 50 incipits (dont 10 avec leur version
distordue) pour tester sans télécharger la base complète.

Démo image → MIDI sur un incipit :

```sh
uv run python -m omr.cli --corpus sample_corpus --id 000051650-1_1_1
```

Évaluation (PA / RA / NA) sur l'échantillon :

```sh
uv run python scripts/benchmark.py --corpus sample_corpus --n 50
```

Sur 50 incipits, les scores sont plus bruités que sur la base complète (le
classifieur de durées s'entraîne sur peu d'exemples) ; c'est un test de bon
fonctionnement, pas la reproduction du tableau du livrable.

## Base complète

Pour reproduire les chiffres du livrable, placer la base CameraPrIMuS dans
`base_1/Corpus/` (non versionnée) et lancer :

```sh
uv run python scripts/benchmark.py --n 800               # durées k-NN (version retenue)
uv run python scripts/benchmark.py --n 800 --geom        # durées géométriques (baseline)
uv run python scripts/benchmark.py --n 400 --key-oracle  # armure oracle
```
