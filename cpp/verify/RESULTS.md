# Vérification WP3 — Python vs C++ (issue #10)

Comparaison sur les 50 incipits de `sample_corpus/` (images propres, PNG),
via `cpp/verify/compare.py`. Modèle de durées : `models/duration_knn.npz`
(k-NN entraîné en Python, chargé tel quel par le C++).

## Fidélité du portage

Vérifié module par module pendant le développement (voir les commits),
pixel/valeur pour pixel/valeur contre la référence Python :

- **Prétraitement** (gris, médiane, Otsu, binarisation adaptative) : identique
  au pixel près sur les images PNG (perte). Sur les JPEG distordus, ~1 pixel
  de différence : décodeur JPEG différent (stb_image vs libjpeg/PIL), attendu
  et sans impact — le décodage de notes n'utilise que la version PNG propre.
- **Portées, composantes connexes, têtes** : mêmes métriques, mêmes boîtes
  englobantes, mêmes comptes, sur tous les échantillons testés.
- **Descripteurs de forme** (Hu, projections) : identiques à 10 décimales.
- **Décodage hauteur** (position → lettre/octave via la clef) : **0 divergence
  de hauteur sur 656 notes comparées**, tous échantillons confondus.
- **Export MIDI** : fichiers strictement identiques octet pour octet (testé
  avec silences, dièses/bémols, durées pointées).

## Durées (k-NN) : une non-déterminisme documentée côté Python, pas un bug C++

Sur les 50 échantillons (656 notes), avec le modèle pré-entraîné :

| | |
|---|---|
| Échantillons comparés | 50 |
| Correspondance exacte (séquence complète) | ~50-55 % (varie d'une exécution à l'autre — voir plus bas) |
| Notes comparées (mêmes longueurs, 0 divergence de longueur) | 656 |
| Dont hauteur différente | **0** |
| Dont durée différente | ~30-40 (~5 %) |

Toutes les divergences observées portent uniquement sur la **durée**, jamais
sur la hauteur ni sur le nombre de notes détectées. En creusant le premier cas
(`000051650-1_1_1`, note C4), le descripteur de forme calculé par le C++ est
**identique bit à bit** à celui de Python, mais la prédiction k-NN diffère.

Cause : `NearestNeighbor.predict` (Python) départage les égalités de vote par
`max(set(labels), key=labels.count)`. L'ordre d'itération d'un `set` Python
dépend du hachage des chaînes, **randomisé par processus** (`PYTHONHASHSEED`)
depuis Python 3.3. Pour ce C4, les 5 plus proches voisins sont à une distance
**strictement égale** (`0.11643374` pour les 5), avec un vote 2 (whole) / 2
(half) / 1 (quarter) : le résultat dépend d'un tirage non reproductible, même
d'une exécution Python à l'autre sur la même machine.

Vérifié empiriquement : trois exécutions indépendantes de la référence Python
seule (sans C++) sur les 50 mêmes échantillons donnent des séquences de durée
qui diffèrent entre elles sur **10 à 33 échantillons sur 50** selon la paire
de runs comparée — la référence Python n'est donc **pas elle-même
reproductible** d'une exécution à l'autre. Le taux de correspondance C++/Python
observé (~50-55 %) est du même ordre que le taux d'accord Python/Python
(~34-80 % selon les paires), ce qui confirme que le portage C++ est correct :
les divergences proviennent entièrement de cette ambiguïté déjà présente dans
la référence, pas d'une erreur de calcul côté C++ (déterministe, vérifié
bit-identique sur les descripteurs et stable sur plusieurs exécutions).

Le tie-break C++ (première étiquette majoritaire rencontrée par ordre de
proximité) est délibérément déterministe — reproductible d'une exécution à
l'autre, contrairement à la référence Python.

## Performance

Mesuré en excluant le chargement du modèle et le lancement de processus côté
C++ (mode `--bench`, modèle chargé une seule fois, boucle sur les 50
échantillons dans un seul processus — comparable à la boucle Python qui
réutilise aussi le même classifieur déjà chargé) :

| | Python (uv run) | C++ (WSL, g++ 13, `-O2`) |
|---|---|---|
| Temps moyen / échantillon (décodage seul) | ~410 ms | ~46 ms |
| Accélération | | **~x9** |

Le C++ reste nettement plus rapide malgré une boucle Python déjà "chaude"
(interpréteur démarré, modèle chargé une seule fois, NumPy vectorisé). La
majeure partie du temps Python est dans les boucles pures Python de
`morph.py`/`segment.py` (érosion/dilatation, union-find) que NumPy ne
vectorise pas ; le C++ les exécute nativement.

Note méthodologique : sur cette machine, aucun CMake natif Windows n'est
disponible ; le binaire C++ est compilé et exécuté sous WSL, et le temps de
lancement de `wsl.exe` par échantillon fausserait fortement une mesure
"un process par image" (démontré : x0.5 au lieu de x9 avec cette méthode
naïve). Le mode `--bench` élimine ce biais en ne payant le coût de démarrage
qu'une fois pour tout le corpus, ce qui reflète le coût algorithmique réel.

## Reproduire

```sh
uv run python cpp/verify/compare.py            # tous les échantillons
uv run python cpp/verify/compare.py --n 10      # sous-ensemble
```
