# Journal de suivi — Projet OMR

Suivi détaillé des étapes réalisées. Sujet 1.11 (analyse automatique de
partitions). Binôme Thibault Bourdoiseau & Léo Li.

---

## WP1 — Prise en main et évaluation (état au démarrage)

### WP1-T1 — État de l'art ✅
- `livrable1_etat_de_lart/main.tex` : 1 page décrivant la chaîne OMR classique
  (prétraitement/binarisation, suppression des portées, segmentation,
  classification, décodage + export MIDI), avec bibliographie.

### WP1-T2 — Méthodes d'évaluation ✅
- `livrable2_evaluation/main.tex` : 3 critères numériques.
  1. **Pitch Accuracy (PA)** — % de hauteurs correctes après alignement des
     séquences par distance d'édition (Levenshtein).
  2. **Rhythm Accuracy (RA) + Note Accuracy (NA)** — % de durées / de notes
     complètes (hauteur+durée) correctes, même alignement.
  3. **F1 de détection** des têtes de notes (IoU ≥ 0.5, appariement glouton).
- `livrable2_evaluation/tools/metrics.py` : `iou_box`, `detection_f1`,
  `note_scores` (NumPy seul, pour portage C++).
- `livrable2_evaluation/tools/evaluate.py` : compare GT/prédiction (JSON).
- `livrable2_evaluation/base.tex` : choix des bases (DeepScoresV2 imprimé,
  MUSCIMA++ manuscrit) + annotation manuelle PA/RA.

**Bases disponibles**
- `base_1/` CameraPrIMuS — 87 678 incipits monophoniques, `.png` propre +
  `_distorted.jpg` + vérité-terrain `.semantic`. → base principale du pipeline.
- `base_2/ds2_dense/` DeepScoresV2 — partitions imprimées, boîtes COCO.
  → critère F1 (secondaire).

---

## WP2 — Prototype Python (en cours)

Décision : pipeline complet sur PrIMuS, **NumPy + Pillow uniquement** (Otsu,
morphologie, composantes connexes réimplémentés à la main → portage C++ direct
au WP3, et conforme à l'esprit TIFO). Environnement géré par **uv**.

### Étape 0 — Échafaudage ✅ (commit `bb5b93d`)
- Dépôt git initialisé, remote GitLab `origin`, branche `main`.
- `.gitignore` : exclut `base_1/`, `base_2/`, `.venv/`, fichiers LaTeX, `out/`.
- `pyproject.toml` (uv) : `numpy`, `pillow`, extra `dev` = `pytest`.
- Structure `omr/` + `tests/`.

### Étape 1 — Prétraitement ✅ (`omr/preprocess.py`)
- `median_blur(img, k)` : filtre médian k×k (empilement de décalages NumPy).
- `otsu_threshold(gray)` : seuil global d'Otsu (maximisation de la variance
  inter-classes sur l'histogramme).
- `binarize(...)` : binarisation Otsu (globale) **ou** adaptative (moyenne
  locale par image intégrale — utile pour le `.jpg` distordu non uniforme).
- `preprocess(...)` : médian puis binarisation ; encre = `True`.
- Validation visuelle : binarisation propre nette ; distordue (adaptatif)
  conserve la portée gondolée avec peu de bruit.

### Étape 2 — Chargement & vérité-terrain ✅ (`omr/io_primus.py`)
- `load_gray(path)` : image → tableau NumPy niveaux de gris.
- `parse_semantic(path)` : `.semantic` → liste `{"pitch", "duration"}`
  (notes + silences ; durées normalisées, fermata retirée, points conservés ;
  clef/armure/mesure/barline/tie/gracenote ignorés).
- `list_samples(corpus_dir, limit)` : itère les échantillons du corpus.
- `PrimusSample` : chemins `.png` / `_distorted.jpg` / `.semantic`.

### Outillage
- `omr/cli.py` : démo (binarise propre + distordu, affiche les notes).
- Tests : `tests/test_preprocess.py`, `tests/test_io_primus.py` — 5/5 OK.

### Étape 3 — Portées ✅ (`omr/staff.py`)
- `estimate_metrics(mask)` : épaisseur des lignes et interligne par mode des
  longueurs de runs verticaux (noir = épaisseur, blanc = interligne).
- `staff_line_rows(mask)` : positions des 5 lignes par projection horizontale
  (pics regroupés). Fiable sur l'imprimé propre.
- `remove_staff_lines(mask, metrics)` : **ouverture morphologique à ES vertical**
  (TIFO2 §5.3) — garde les structures plus hautes que l'épaisseur d'une ligne →
  enlève les lignes tout en gardant hampes, têtes et **ligatures**.
  `remove_staff_lines_runlength` : variante par runs verticaux (résultat
  identique). `staff_lines_mask` : isole les lignes par ouverture à ES horizontal.
- Choix de l'ES **vertical** plutôt qu'horizontal : l'ES horizontal (texte
  littéral de l'état de l'art) abîme les ligatures (horizontales et longues),
  vérifié visuellement. Écart documenté dans `CORRESPONDANCE_COURS.md`.
- Résultats : propre → interligne 15, épaisseur 3, 5 lignes détectées ;
  suppression nette sur propre **et** distordu (validation visuelle).
- **Limite connue** : `staff_line_rows` (projection horizontale) ne trouve
  qu'une ligne sur la portée gondolée du `.jpg` distordu ; le décodage des
  hauteurs sur images distordues demandera Hough / chemins stables (cf. état
  de l'art). N'affecte pas la suppression des lignes.

### Étape 4 — Segmentation ✅ (`omr/segment.py`)
- `label_components(mask)` : étiquetage en composantes connexes 8-connexité
  réimplémenté (union-find à deux passes, NumPy pur → portable C++).
- `components_from_labels` / `Component` : boîtes englobantes + aire.
- `segment(mask, min_area)` : filtre les petites composantes (bruit) et trie
  les symboles de gauche à droite. `min_area` réglé sur épaisseur×interligne.
- Résultats : propre → 21 symboles, distordu → 39 (plus fragmenté).
- **Observation clé (attendue)** : les groupes **ligaturés** (têtes+hampes+
  ligature reliées) forment une seule composante. Clef, altérations, mesure et
  notes isolées sont bien séparées. → l'étape 5/6 devra **découper les groupes
  ligaturés** en têtes individuelles (morpho/template de tête ovale).

### Étape 5a — Détection des têtes ✅ (`omr/morph.py`, `omr/heads.py`)
- `morph.py` : morphologie mathématique NumPy pur (réutilisable, portable C++) :
  `disk`, `rect`, `erode`, `dilate`, `opening`, `closing`.
- `heads.py` :
  - `fill_holes(mask)` : bouche les trous (têtes vides blanche/ronde → pleines)
    via étiquetage du fond et suppression des composantes touchant le bord.
  - `detect_heads(mask, metrics)` : ouverture morphologique par un disque de
    rayon ≈ 0,4×interligne → ne survivent que les blobs pleins (têtes) ;
    élimine hampes (fines), ligatures (fines verticalement), résidus de lignes.
    Filtrage par taille (≈ interligne), allongement et taux de remplissage.
    Détecte les têtes **y compris dans les groupes ligaturés**.
- Résultats : 15 têtes détectées (12 notes GT) — **bon rappel** (toutes les
  vraies têtes), léger sur-comptage (gracenote exclue du GT + artefacts de
  ligature). Suffisant : le décodage aligne par distance d'édition, et le F1
  strict sera mesuré sur DeepScores (vraies boîtes).
- **À affiner plus tard** : précision (filtrer gracenotes/artefacts), têtes
  vides vs pleines pour la durée.

### Étape 5b — Descripteurs + plus proche voisin ✅ (briques) ; clef ✅
- `descriptors.py` : **moments de Hu** (7, invariants translation/échelle/
  rotation), **histogrammes de projection** (lignes/colonnes rééchantillonnés),
  ratios allongement/remplissage. → MLRF L1 (descripteurs OCR).
- `knn.py` : `NearestNeighbor` (k-NN, distances euclidiennes standardisées).
  → MLRF L3 (matching/plus proche voisin).
- `dataset.py` : extraction d'un symbole (composante la plus à gauche et haute
  = clef), prétraitement + suppression portées + segmentation + description.
- `scripts/clef_experiment.py` : auto-étiquetage par le 1er token `.semantic`
  (toujours une clef), split 70/30.
- **Résultats clef (7 classes)** : imprimé propre **99,2 %** (n=800),
  distordu **91,7 %**. Valide la chaîne descripteurs + k-NN sur données réelles.
- **À faire en 5b** : silences et altérations (besoin d'étiquetage par position
  dans la séquence), tête pleine vs vide (pour la durée).
- **Perf** : `label_components` (Python pur, double boucle) est lent (~0,1 s/img)
  → à vectoriser ou porter en C++ (WP3).

### Étapes suivantes (à faire)
- [ ] **Étape 5b (suite)** — silences, altérations, tête pleine/vide.
- [ ] **Étape 6** — `decode.py` : hauteur (position/portée + clef) + durée
  (forme), séquence de notes, export MIDI, branchement sur `evaluate.py`.

---

## WP3 — Implémentation C++ (à venir)

## WP4 — Pilotage
- Remote GitLab : `https://gitlab.cri.epita.fr/leo.li/reconnaissance.git`.
- À mettre en place : branches dev/release, issues, CI, GitLab Pages.
