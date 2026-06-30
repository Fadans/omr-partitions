# Correspondance prototype ↔ notions de cours

Le prototype mobilise **uniquement des notions vues en TIFO 1, TIFO 2 et MLRF**
(pas de réseaux de neurones). Tableau de traçabilité pour la soutenance.

| Module / fonction | Notion mobilisée | Cours |
|---|---|---|
| `preprocess.median_blur` | Filtre médian (bruit sel & poivre) | TIFO1 ch5 §2.4 ; ch6 §5.2 |
| `preprocess.otsu_threshold` | Binarisation, critère d'Otsu | TIFO1 ch4 §9.4 |
| `preprocess.binarize` (adaptatif) | Image intégrable (summed area table) pour moyenne locale | TIFO1 ch7 §2.3 |
| `preprocess` (gris) | Conversion RGB → niveaux de gris | TIFO1 ch3 §2.3 |
| `staff.estimate_metrics` | Longueurs de runs (épaisseur/interligne) | TIFO1 ch3 §3 ; état de l'art |
| `staff.staff_line_rows` | Projection / histogramme de lignes | TIFO1 ch4 ; état de l'art |
| `staff.staff_lines_mask` | Ouverture morphologique (ES horizontal) | TIFO2 §5.3 |
| `staff.remove_staff_lines` | **Ouverture morphologique** (ES vertical : garde les structures plus hautes que l'épaisseur d'une ligne ⇒ enlève les lignes, garde têtes/hampes/ligatures) | TIFO2 §5.3 |
| `staff.remove_staff_lines_combined` | Union ouverture morphologique + suppression « pont » (conserve les pixels où un symbole traverse la ligne) ⇒ préserve aussi les contours fins (têtes creuses, altérations) | TIFO2 §5.3, connexité |
| `morph` (erode/dilate/opening/closing) | Les 4 opérateurs morphologiques fondamentaux | TIFO2 §5 |
| `morph` (disk/rect) | Élément structurant | TIFO2 §4.2 |
| `segment.label_components` | Connexité, étiquetage en composantes connexes (8-conn.) | TIFO1 ch3 §3.4 ; TIFO2 §4.1 |
| `heads.fill_holes` | Reconstruction géodésique (bouchage de trous) | TIFO2 §11 |
| `heads.detect_heads` | Granulométrie / ouverture par disque (sélection de blobs ronds) | TIFO2 §12 ; §5.3 |
| `classify` (étape 5b, à venir) | Descripteurs de forme OCR : moments de Hu, histogrammes de projection, descripteurs de Fourier | MLRF L1 (descripteurs OCR) |
| `classify` — décision | Plus proche voisin / matching (distance) | MLRF L3 (matching, bruteforce/kD-tree) |
| `key_signature` / lecture armure | Descripteur de bande (projections) + k-NN ; théorie musicale (ordre fixe des altérations) | MLRF L1/L3 |
| `metrics.note_scores` | Distance d'édition (Levenshtein) pour l'alignement | MLRF L1 (note 1) |
| `metrics.detection_f1` | IoU (indice de Jaccard), précision/rappel/F1 | MLRF (évaluation) |

## Notions de cours mobilisables en réserve (pistes d'amélioration)
- **Top-hat** (TIFO2 §7) : extraction de symboles plus petits que l'ES.
- **Watershed sur carte des distances** (TIFO2 §13, « pour objets ronds ») :
  séparation de têtes qui se touchent (accords) — non nécessaire en monophonique.
- **Squelette / carte des distances** (TIFO2 §9-10) : analyse hampes/ligatures.
- **Détecteurs de coins (Harris)** / **descripteurs locaux (HoG, SIFT/ORB)**
  (MLRF L2) : appariement de symboles complexes si besoin.
- **Homographie** (MLRF L3) : redressement perspectif des photos distordues.

## Écart documenté vs état de l'art (livrable 1)
L'état de l'art décrit la suppression des portées par « ouverture morphologique
avec un élément structurant **horizontal** puis soustraction ». Testée, cette
variante **abîme les ligatures** (elles aussi horizontales et longues). Nous
utilisons donc une ouverture par élément structurant **vertical**, qui reste une
ouverture morphologique (même notion, TIFO2 §5.3) mais préserve les ligatures.
