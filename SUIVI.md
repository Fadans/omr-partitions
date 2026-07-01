# Journal de suivi — Projet OMR

Suivi détaillé des étapes réalisées. Sujet 1.11 (analyse automatique de
partitions). Binôme Thibault Bourdoiseau & Léo Li.

---

## ⚠️ À ne pas oublier (actions / soutenance)
- **Suppression des portées — ES vertical vs horizontal** : l'état de l'art
  (livrable 1) écrit « ouverture morphologique avec un ES **horizontal** ». Le
  prototype utilise un ES **vertical** (l'ES horizontal abîme les ligatures,
  démontré visuellement). Reste une « ouverture morphologique » (TIFO2 §5.3).
  → **À mentionner en soutenance** comme amélioration justifiée, et **corriger
  le mot dans l'état de l'art si encore éditable**. Détail dans
  `CORRESPONDANCE_COURS.md`.

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
  `remove_staff_lines_runlength` : variante par runs verticaux (identique).
  `staff_lines_mask` : isole les lignes par ouverture à ES horizontal.
- `remove_staff_lines_bridge` : n'efface une ligne **que là où aucun symbole ne
  la traverse** (pixels « pont » conservés si encre au-dessus ET au-dessous) →
  préserve les contours fins (têtes creuses, altérations) mais abîme les
  ligatures coïncidant avec une ligne.
- **`remove_staff_lines_combined`** (méthode du pipeline) : union de l'ouverture
  morphologique et du pont → garde ligatures/notes pleines (morpho) **et** têtes
  creuses/altérations (pont) ; une vraie ligne (enlevée par les deux) disparaît.
  C'est ce qui a débloqué la détection des têtes creuses.
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

### Étape 6 — Décodage + MIDI + évaluation bout-en-bout ✅ (`decode.py`, `midi.py`)
- `decode.py` :
  - **Hauteur** : géométrie de portée (`staff_line_rows` sur le masque AVEC
    lignes → ligne du bas + demi-pas), clef → pitch de référence et ligne
    (`clef_reference`), position de la tête → nom + octave diatonique.
  - **Durée** : tête pleine/vide (remplissage), détection de hampe
    (`_find_stem`, plus long run vertical près de la tête), **comptage des
    ligatures/crochets** (`_count_beams` : runs verticaux dans la zone de débord
    de la hampe) → ronde/blanche/noire/croche/double…
  - `clean_heads` : filtre par hauteur médiane + suppression des chevauchements
    (enlève gracenotes/fragments/doublons).
- `midi.py` : **writer Standard MIDI File maison** (pur Python, zéro dépendance
  → portable C++) : `pitch_to_midi`, `duration_to_ticks` (points gérés),
  `write_midi`.
- `scripts/decode_experiment.py` : PA/RA/NA via `note_scores` (livrable 2),
  clef supposée connue (reconnue à 99 % par ailleurs), + PA diatonique.
- **Résultats (imprimé propre, n=120)** : PA diatonique **73 %** (médiane 85 %),
  PA stricte 57 %, RA 31 %, NA 24 %. Sur partition simple (croches/noires) le
  décodage des hauteurs est ~parfait ; la médiane (85 %) montre que la moyenne
  est tirée vers le bas par les incipits en notes creuses (cf. limites).
- **Bugs corrigés** : géométrie calculée par erreur sur le masque sans lignes ;
  `fill_holes` sur-remplissait les régions fermées des groupes ligaturés
  (→ remplissage limité aux petits trous).

### Étape 5b (suite) — Lecture de l'armure (altérations) ✅ théorie / ⚠️ détection
- `key_signature.py` : table tonalités majeures PrIMuS → (type, nombre), ordre
  fixe des altérations (dièses F C G D A E B ; bémols B E A D G C F),
  `alteration_map`, `apply_alteration`. Astuce : compter + typer suffit (l'ordre
  est fixe). Branché dans `decode(..., alt_map=...)`. Tests unitaires.
- **Plafond mesuré (armure = oracle vérité-terrain, n=60)** : PA stricte
  **57,5 % → 71,2 %** (plafond diatonique 74,4 %), NA 21,6 → 27,8 %.
  → confirme que lire l'armure récupère presque tout l'écart strict/diatonique
  et valide le module bout-en-bout.
- **Détection automatique** (`dataset.keysig_strip_descriptor` : descripteur de
  la bande clef→1re note = projections + largeur + encre + nb colonnes ;
  `scripts/key_experiment.py`, k-NN sur la tonalité) : **~50 % type, ~40 %
  nombre** — insuffisant pour l'intégrer sans risquer de dégrader la PA.
  **Bloqueur** : la suppression de portée fragmente les traits fins (dièses,
  bémols), comme pour les têtes creuses. Correctif amont nécessaire (suppression
  préservant les contours / reconstruction de symboles) avant d'activer la
  détection automatique dans le pipeline.

### Banc d'essai PA/RA/NA (`scripts/benchmark.py`)
« simple » = sans armure et sans notes creuses. **Split train/test disjoint**
(70/30) : le classifieur de durées est entraîné sur la partie *train*,
l'évaluation ne porte que sur la partie *test*. Table principale : n=800
(test=240) ; rappel géométrique mesuré à n=400 (test=120).

| Sous-ensemble | N | PA strict | PA diat. | RA | NA |
|---|---|---|---|---|---|
| **Durées géométriques (ancien, n=400)** | | | | | |
| PrIMuS simple | 19 | 65,9 | 67,6 | 37,8 | 35,4 |
| PrIMuS général | 120 | 56,4 | 75,0 | 35,4 | 26,0 |
| **Durées k-NN (nouveau, défaut, n=800)** | | | | | |
| PrIMuS simple | 61 | 83,5 | 85,5 | **77,3** | **74,1** |
| PrIMuS général | 240 | 65,9 | 83,5 | **71,2** | **56,1** |
| PrIMuS général (armure oracle, n=400) | 120 | 82,1 | 84,3 | 69,4 | **66,2** |

**Gain majeur — classifieur de durées k-NN** (`omr/duration_knn.py`). Analyse
de la matrice de confusion des durées (`scripts/diag_full.py`) : les règles
géométriques (`_count_beams`) confondaient massivement noire↔croche et
double-croche↔croche (à cette résolution une ligature fait ~3 px). Remplacées
par un **descripteur de forme (projections + géométrie) de la région
tête+hampe+ligature classé au k-NN** (cohérent avec clef/armure ; MLRF L1/L3).
*Cohérence état de l'art* : notre livrable 1 §4 prévoit explicitement de décrire
les symboles « par quelques mesures caractéristiques […] moments de Hu, puis
[de confier] le classement à un algorithme simple tel que les k plus proches
voisins ». Le livrable 1 §5 décrivait la durée par voie **géométrique** (tête
pleine/vide, hampe, ligatures) ; on **étend** donc la méthode k-NN déjà prévue
(pour clef/silences/altérations) à la durée, la géométrie seule s'étant révélée
peu fiable. Ce n'est pas une méthode étrangère au projet, mais celle déjà
retenue, appliquée à une catégorie de plus.
Sur le même jeu de test (notes bien alignées) la précision de durée passe de
**39 % → 79 %** ; bout-en-bout **RA 35 → 69**, **NA 26 → 51** (doublée).
La PA monte aussi (56 → 64) : des durées correctes améliorent l'alignement par
distance d'édition. Un masque de ligatures par **ouverture morphologique
horizontale** (`decode.beam_mask`, TIFO2 §5.3) fournit un descripteur robuste.
Modèle entraîné/sauvé par `scripts/train_duration.py` → `models/duration_knn.npz`
(chargé par défaut dans `omr/cli.py`).

**Plafond restant = l'armure.** L'oracle montre **+15 NA** (51 → 66) : c'est
désormais le levier n°1, car le rythme étant réglé, corriger les altérations se
convertit directement en NA. Voir limites ci-dessous.

**Historique des gains antérieurs** :
- *Suppression de portée combinée* (cf. étape 3) : PA diatonique général
  77,9 → **80,2** (têtes creuses enfin détectées) ; détection du **type**
  d'armure 51 → 67 %.

#### Journal détaillé de la session d'amélioration (essais gardés / abandonnés)
Démarche : diagnostic d'abord (`scripts/diag_full.py` — matrice de confusion des
durées + décomposition pitch/rythme), puis itérations mesurées. Ordre réel :

1. **Diagnostic** : constat que **NA est plafonnée par le rythme** (RA ≈ 43 ≪
   PA ≈ 60 ; une note juste exige pitch ET durée). La confusion des durées
   révèle un biais massif vers la croche : **noire→croche 269**, double-croche→
   croche 293, blanche→croche/double 90. Cause : `_count_beams` déclarait une
   ligature dès qu'il y avait un peu d'encre latérale (défaut `beams=1`).
2. **Essai A — contiguïté stricte** (ABANDONNÉ) : n'accepter la ligature que si
   l'encre est contiguë à la hampe (arrêt au 1er trou) + extension min 0,5u.
   Résultat : **RA 42,7 → 32,2** — cela tuait les **vraies** ligatures/crochets
   (croche 86,8 → 58,5 %, double-croche 43,6 → 18,4 %), car à ~3 px d'épaisseur
   le scan colonne-par-colonne a des trous. → **reverté**.
3. **Essai B — porte de couverture** (ABANDONNÉ) : exiger qu'une fraction ≥ 40 %
   des colonnes scannées portent une ligature. Résultat : **RA inchangée
   (42,7)** — l'encre parasite (ligature d'un groupe **voisin** captée par la
   portée de scan de 1,6u) couvre elle aussi > 40 %. → **remplacé**.
4. **Essai C — masque de ligatures par ouverture morphologique horizontale**
   (`decode.beam_mask`, GARDÉ comme brique) : un ES horizontal isole les traits
   longs (ligatures), efface hampes fines et têtes rondes (TIFO2 §5.3). Le
   comptage devient plus propre : **RA 42,7 → 43,9**, double-croches 43,6 →
   57,1 %. Mais **noire→croche persiste (266)** : à cette résolution la
   géométrie fine reste fragile (sur- ET sous-comptage, cf. `debug` supprimé).
5. **Pivot — classifieur k-NN de durées** (GARDÉ, méthode par défaut) : voir
   ci-dessus. `beam_mask`/`_count_beams` de l'essai C ne sont **pas jetés** :
   ils restent (a) repli géométrique quand aucun modèle n'est fourni
   (`decode(..., dur_clf=None)`), (b) une **feature** du descripteur k-NN.
6. **Armure automatique** (testée, NON intégrée) : mesurée avant ET après le
   k-NN (`scripts/benchmark_autokey.py`) → apport réel +0,3 à +0,5 NA seulement,
   pour un oracle à +8 puis +16. Diagnostic du comptage d'altérations
   (`diag_keycount`, depuis retiré) : **17 %** de comptage exact, sous-comptage
   systématique. → **laissée désactivée** (une armure fausse dégrade la NA) ;
   plafond oracle documenté (limites ci-dessous).
7. **Correctif `leftmost_tall`** (GARDÉ) : la clef était parfois une barre de
   mesure fine (haute mais étroite). Ajout d'un filtre de largeur (≥ 0,5u).
   N'affecte que les expériences clef/armure — le décodage principal prend la
   clef dans la vérité-terrain (`toks[0]`).
8. **Essai — suppression des fausses têtes dans la zone de la clef** (ABANDONNÉ) :
   écarter les têtes dont le centre est à gauche du bord droit de la clef (une
   note ne peut y être). Mesure : **NA générale 51,4 → 50,9** (léger recul) — le
   détecteur de bord de clef supprime parfois de **vraies premières notes**
   (dégât collatéral > bénéfice). → reverté.
9. **Essai — détection des têtes par corrélation de gabarit** (ABANDONNÉ pour le
   défaut ; gardé comme expérience) : gabarit elliptique glissé + maxima locaux
   (`detect_heads_template`, `scripts/benchmark_heads.py`). Mesure : **ratio
   têtes/GT 2,1–3,0** (vs **1,07** en morphologie) → **NA 51 → 33**. Un gabarit
   plein corrèle avec toute zone pleine (ligatures, clef, jonctions) ; il est
   moins discriminant que l'ouverture morphologique + filtre de forme sur la
   composante entière. **La morphologie actuelle reste supérieure.**
10. **Essai — suppression de portée par chemins stables** (ABANDONNÉ pour le
    défaut ; gardé comme expérience — méthode **annoncée** dans l'état de l'art
    §2, Cardoso 2009). Plus court chemin gauche→droite suivant l'encre
    (programmation dynamique) pour suivre chaque ligne, puis retrait des runs
    fins le long du chemin avec pont (`remove_staff_lines_stable_paths`,
    `scripts/benchmark_staff.py`). Mesure (n=250) : **NA 55,9 (combined) vs 54,8
    (stable paths)**, PA_dia 88,7 vs 89,2 — **équivalent** (stable paths même un
    peu en dessous, dans le bruit). Raison : PrIMuS est **imprimé et droit**,
    l'avantage des chemins stables (lignes courbes/inclinées) ne s'exprime pas ;
    `combined` gère déjà les lignes droites, et la fragmentation ligne↔symbole
    est la même. → n'aide **pas** l'armure/les silences sur du imprimé droit.
    **Confirmation sur images DÉFORMÉES** (Camera-PrIMuS, `--distorted`, portées
    courbes) : les chemins stables **améliorent la lecture des hauteurs**
    (PA_dia **38,3 → 43,2**, PA 29,4 → 31,5) — l'avantage annoncé (suivre des
    lignes courbes/inclinées) **s'exprime bien**. La NA globale reste basse
    (~16 %) car le pipeline sur déformé est bridé ailleurs (binarisation
    adaptative, détection de têtes). **Bilan** : défaut `combined` sur imprimé ;
    chemins stables **pertinents et validés** pour le cas déformé/manuscrit
    (comme prévu dans l'état de l'art §2).

### Silences (rests) — méthode implémentée, opt-in (`omr/rest_knn.py`, `omr/rests.py`)
Demandé par le sujet (« notes, clefs et silences »). Prévalence mesurée
(n=3000) : **65 %** des incipits ont un silence/multi-pause, **39 %** ont un
silence *au milieu* (entre 1re et dernière note, ~2391 au total), **41 %**
commencent par un silence/multi-pause.

**Méthode** (conforme état de l'art §4-5, MLRF) : détecteur géométrique de
candidats (`detect_rests`) + **classifieur k-NN de forme** (`RestClassifier`,
descripteur projections + Hu + géométrie + position verticale relative), en
mode **haute précision** (voisins unanimes sinon « none »). Intégré à `decode`
via le paramètre `rest_clf` (fusion notes+silences par position x), **désactivé
par défaut**. Entraînement : `scripts/train_rest.py` ; éval :
`scripts/rest_class_experiment.py`.

**Résultat : même mur que l'armure.** L'auto-étiquetage n'exploite que les
échantillons où `#candidats == #silences`, or `detect_rests` **sur-détecte** :
count-match **11 %** (18 % avec centrage vertical + exclusion des hampes). D'où
très peu de positifs d'entraînement (40 à n=600, 74 à n=1500) et une détection
médiocre (**précision 6–25 %**). Contaminants non filtrables géométriquement :
fragments de ligatures entre notes beamées, altérations *inline*, points — eux
aussi centrés sur la portée. Une détection aussi bruitée **dégraderait** la
sortie MIDI → **laissée opt-in/OFF**, comme l'auto-armure.
- **Pistes réelles (non tenues, hors budget WP2)** : corrélation de gabarit de
  silence (état de l'art §4, contourne la sur-détection) ; meilleure isolation
  des composantes (analyse hampes/ligatures) pour débloquer les positifs ;
  gérer les silences/multi-pauses **en tête** (41 %, hors champ actuel).

### Limites connues (honnêtes, à améliorer)
- **Altérations / armure = plafond n°1, CUL-DE-SAC avéré (méthodes classiques)** :
  théorie + intégration faites (l'oracle donne **NA 51 → 66**, PA stricte
  64 → 82). Mais **aucune** détection automatique n'atteint une précision
  *nette-positive*. Comptage des altérations mesuré sur 4 approches distinctes :
  - k-NN holistique sur profil de projection du strip : **33 %** (puis 21 % en
    ré-mesure avec correctif clef, beaucoup d'échantillons rejetés) ;
  - regroupement géométrique en x des composantes : **17 %** ;
  - fermeture morphologique + composantes connexes : **15 %**.

  Tous **sous la baseline « toujours 0 »** (~40 % de CM). Bout-en-bout l'auto
  n'apporte que +0,3 NA (`scripts/benchmark_autokey` : none 53,1 → auto 53,4 →
  oracle 69,1). Trois causes cumulées, diagnostiquées : (1) la suppression de
  portée **fragmente** les symboles fins (dièse/bémol ≈ 3 px de trait, cassés en
  plusieurs morceaux) ; (2) des **fausses têtes** près de la clef corrompent la
  bande ; (3) la bande clef→1re note contient aussi la **métrique,
  multi-pauses, barres et silences**. Une armure *fausse* (≈ 75 % du temps)
  dégraderait la NA sur toutes les notes de l'incipit. → l'auto reste
  **désactivé** ; l'oracle documente le gain **théoriquement** atteignable.
  Piste réelle (non tentée) : ré-analyser l'armure sur une image **sans
  suppression de portée** (les altérations y seraient entières mais collées aux
  lignes) ou par corrélation de gabarit robuste — hors budget WP2.
- **Têtes creuses** (blanches/rondes) : nettement améliorées par la suppression
  combinée (cf. étape 3) ; restent un peu moins fiables que les têtes pleines.
- **Durées** : nettement améliorées par le k-NN (RA 35 → 69). Restent limitées
  par les blanches/rondes (peu d'exemples d'entraînement) et les notes pointées
  (durée de base seulement).
- **Silences** : méthode implémentée (candidats + k-NN de forme, opt-in) mais
  détection non fiable (même mur que l'armure : sur-détection des candidats,
  count-match 11–18 %). Décodés uniquement si `rest_clf` fourni. Cf. section
  dédiée.

### Étapes suivantes (à faire)
- [ ] **Armure automatique (levier n°1, +15 NA)** : corriger fausses têtes près
  de la clef, isoler la bande d'armure (s'arrêter à la métrique), compter les
  altérations de façon robuste à la fragmentation (fusion de fragments, ou
  classif par composante dièse/bémol/autre au k-NN).
- [ ] **Silences** — débloquer les positifs d'entraînement (meilleure détection
  de candidats ou corrélation de gabarit) pour activer `rest_clf` par défaut.
- [ ] **Durées** — plus de données d'entraînement (corpus = 87 k incipits),
  gérer les notes pointées.
- [ ] **WP3** — portage C++ ; **WP4** — CI, GitLab Pages, issues.

---

## Bilan WP2 — réalisé vs annoncé (état de l'art / sujet)

Le sujet demande : lire **notes (hauteur + durée), clefs et silences**, export
MIDI, avec les seules notions TIFO/MLRF (pas de réseau). L'état de l'art
(livrable 1) décrit une chaîne : prétraitement → portées → détection de symboles
→ classification (descripteurs + k-NN/SVM) → décodage → MIDI. Correspondance :

| Étape annoncée (livrable 1) | Réalisé | État |
|---|---|---|
| Prétraitement (gris, Otsu, médian, adaptatif) | `preprocess.py` | ✅ |
| Métriques de portée (run-length) | `staff.estimate_metrics` | ✅ |
| Suppression des portées (ouverture morpho.) | `staff.remove_staff_lines_combined` (ES **vertical** + pont, cf. écart) | ✅ |
| Segmentation (composantes connexes) | `segment.py` (union-find 8-conn) | ✅ |
| Détection des têtes (gabarit **ou** morpho. ovale) | `heads.detect_heads` (morpho.) ; gabarit testé et écarté | ✅ (morpho.) |
| Clef (descripteurs + k-NN) | `dataset` + `knn` (exp.) ; décodage prend la clef GT | ⚠️ classif prototypée, non câblée |
| **Hauteurs** (position + clef) | `decode.head_pitch` | ✅ |
| **Durées** (tête/hampe/ligatures/points) | `duration_knn` (k-NN) + repli géométrique | ✅ (gain majeur) |
| **Altérations / armure** (descripteurs + k-NN + théorie) | `key_signature` + `dataset` ; théorie ✅, **auto = mur** | ⚠️ oracle OK, auto OFF |
| **Silences** (descripteurs + k-NN) | `rest_knn` (opt-in) | ⚠️ méthode OK, détection = mur |
| Export MIDI | `midi.py` (SMF pur) | ✅ |
| Évaluation (PA/RA/NA, IoU/F1) | `livrable2_evaluation` + benchmarks | ✅ |

**Chiffres finaux (test disjoint n=800)** : simple NA **74**, général NA **56**
(PA diat. 84, RA 71) ; général + armure oracle NA **66**.

**Écarts assumés et documentés** (pour la soutenance) :
1. Suppression de portée par ES **vertical** (l'horizontal de l'état de l'art
   abîme les ligatures) — reste une ouverture morphologique.
2. Durées par **k-NN** (annoncé géométrique au §5) : extension de la méthode
   k-NN déjà prévue au §4, car la géométrie seule était peu fiable.
3. **Armure** et **silences** : méthodes conformes implémentées, mais détection
   automatique non fiable (fragmentation + sur-détection à cette résolution) →
   laissées OFF/opt-in, plafonds documentés. Ce sont les vraies limites d'un
   prototype **sans réseau de neurones**.
4. Détection de têtes : les **deux** options du §4 testées ; morphologie retenue.

**Honnêteté méthodo.** Tous les essais (gardés/abandonnés) sont journalisés
ci-dessus (essais 1→9). Aucune amélioration annoncée sans mesure ; les reculs
(contiguïté, couverture, filtre clef, gabarit têtes) sont documentés et revertés.

---

## WP3 — Implémentation C++ (à venir)

## WP4 — Pilotage
- Remote GitLab : `https://gitlab.cri.epita.fr/leo.li/reconnaissance.git`.
- À mettre en place : branches dev/release, issues, CI, GitLab Pages.
