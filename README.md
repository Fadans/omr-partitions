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

## Lancer le prototype

Démo image → MIDI sur un incipit :

```sh
uv run python -m omr.cli --id <incipit>
```

Évaluation (PA / RA / NA) sur `n` incipits :

```sh
uv run python scripts/benchmark.py --n 800               # durées k-NN (version retenue)
uv run python scripts/benchmark.py --n 800 --geom        # durées géométriques (baseline)
uv run python scripts/benchmark.py --n 400 --key-oracle  # armure oracle
```

La base CameraPrIMuS doit être présente dans `base_1/Corpus/` (non versionnée).
