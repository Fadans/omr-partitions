# Bilan WP2 — Prototype OMR (Python)

Analyse automatique de partitions musicales monophoniques (sujet 1.11).
Uniquement notions TIFO 1 / TIFO 2 / MLRF — **aucun réseau de neurones**.
Binôme Thibault Bourdoiseau & Léo Li.

## 1. Métriques d'évaluation (livrable 2)

Alignement de la séquence prédite sur la vérité-terrain par **distance d'édition
(Levenshtein)** sur les couples `(hauteur, durée)`, puis :

- **PA — Pitch Accuracy** : % de notes à la **hauteur** correcte.
  - *stricte* : avec altérations (dièse/bémol) ; *diatonique* : altérations ignorées.
- **RA — Rhythm Accuracy** : % de notes à la **durée** correcte.
- **NA — Note Accuracy** : % de notes correctes **sur les deux** (hauteur ET durée).
- **F1 détection (IoU / Jaccard)** : précision/rappel/F1 sur les boîtes, pour les
  tâches de détection. *PrIMuS n'a pas de coordonnées pixel* → on utilise comme
  proxy le **ratio têtes détectées / notes réelles** (idéal = 1,0).

Sous-ensembles : **simple** (sans armure ni notes creuses) et **général** (tout).
Split train/test disjoint 70/30 ; évaluation sur le test uniquement.

## 2. Comparaison V1 → V2 (n=600, test=180, mêmes conditions)

**V1** = prototype complet initial (durées par règles géométriques de comptage
de ligatures). **V2** = version finale après décisions (durées par classifieur
**k-NN** sur descripteurs de forme).

| Sous-ensemble | Version | PA stricte | PA diat. | RA | NA |
|---|---|---|---|---|---|
| **simple**  | V1 | 62,4 | 64,4 | 28,3 | 26,1 |
| **simple**  | **V2** | **72,4** | **74,4** | **65,0** | **60,7** |
| **général** | V1 | 52,8 | 66,7 | 32,3 | 23,2 |
| **général** | **V2** | **60,3** | **74,1** | **64,1** | **50,2** |
| général | V2 + armure oracle | 73,3 | 74,4 | 67,0 | **61,8** |

**Gains V1 → V2 (général)** : PA +7,5 · PA diat. +7,4 · RA **+31,8** · NA **+27,0**
(NA **plus que doublée**, 23 → 50). La PA monte aussi : des durées correctes
améliorent l'alignement par distance d'édition.

### Métriques annexes mesurées
- **Classification de durée** (sur notes bien alignées) : géométrique **39 %** →
  k-NN **79 %**.
- **Détection de têtes** (proxy ratio têtes/notes) : morphologie **~1,1** ;
  corrélation de gabarit testée **~2,1–3,0** (sur-détection) → morphologie retenue.
- **Détection de portées sur images déformées** (Camera-PrIMuS) : projection
  trouve les 5 lignes dans **61 %** des cas, **Hough dans 99 %** ; suppression
  par **chemins stables** améliore la lecture des hauteurs (PA diat. **38 → 43**).

## 3. Décisions finales retenues

- **Durées : k-NN** sur descripteur de forme (projections + Hu + géométrie) de la
  région tête+hampe+ligature. *Le gros gain.* Repli géométrique conservé.
- **Suppression de portées : ouverture morphologique + « pont »** (`combined`)
  par défaut (imprimé) ; **chemins stables** validés pour le déformé/manuscrit.
- **Détection de lignes : projection** par défaut ; **Hough** pour l'incliné.
- **Détection de têtes : morphologie** (ouverture par disque + filtre de forme).
- **Armure & silences : méthodes implémentées mais détection auto non fiable**
  (fragmentation des symboles fins à cette résolution) → **désactivées / opt-in**,
  plafonds documentés (armure oracle : NA générale 50 → 62).

## 4. Fonctionnement du prototype (chaîne de traitement)

Image de partition → séquence de notes → fichier MIDI.

1. **Prétraitement** (`preprocess.py`) : niveaux de gris → **filtre médian**
   (bruit) → **binarisation d'Otsu** (imprimé) ou **seuil adaptatif par image
   intégrale** (photos déformées). TIFO 1.
2. **Portées** (`staff.py`) : mesure de l'**épaisseur de ligne** et de
   l'**interligne** par longueurs de runs ; localisation des 5 lignes par
   **projection** (ou **Hough** si incliné) ; **suppression** des lignes par
   ouverture morphologique verticale + pont (ou **chemins stables** si courbe),
   en préservant les symboles qui traversent. TIFO 1/2.
3. **Segmentation** (`segment.py`) : **composantes connexes** (union-find,
   8-connexité) → un composant par symbole. TIFO 1/2.
4. **Détection des têtes** (`heads.py`) : bouchage des petits trous
   (**reconstruction géodésique**) → **ouverture par disque** (granulométrie) →
   filtrage par taille / allongement / remplissage. TIFO 2.
5. **Décodage des hauteurs** (`decode.py`) : géométrie de la portée (ligne du bas
   + demi-interligne) → position verticale de chaque tête → lettre + octave via la
   **référence de la clef**.
6. **Décodage des durées** (`duration_knn.py`) : masque de ligatures par
   **ouverture horizontale** ; descripteur de forme de la région tête+hampe+
   ligature → **k-NN** → ronde/blanche/noire/croche/double/… MLRF.
7. **Armure** (`key_signature.py`, optionnel) : type + nombre d'altérations →
   application aux hauteurs selon l'ordre fixe (FA#DO#SOL#… / SIb MIb LAb…).
8. **Silences** (`rest_knn.py`, opt-in) : candidats géométriques + **k-NN** de
   forme, fusionnés aux notes par position horizontale.
9. **Export MIDI** (`midi.py`) : hauteur → numéro MIDI, durée → tics ; écriture
   d'un fichier Standard MIDI pur.
10. **Évaluation** (`livrable2_evaluation/`) : alignement par distance d'édition
    → PA / RA / NA.

La traçabilité complète module ↔ notion de cours est dans `CORRESPONDANCE_COURS.md` ;
le journal détaillé des essais (gardés/abandonnés) dans `SUIVI.md`.

## 5. Limites connues (honnêtes)

- **Armure & silences** : détection automatique non fiable (fragmentation +
  sur-détection à ~3 px de trait) ; c'est la limite structurelle d'un prototype
  **sans réseau**. Plafonds chiffrés (oracle).
- **Durées fines / notes pointées** : durée de base seulement.
- **Images déformées** : pipeline global encore faible (binarisation, têtes),
  même si Hough + chemins stables améliorent l'étape portées.

Contexte littérature (`references.bib`) : en OMR **classique**, une exactitude
note-level de 50–80 % sur données réelles est attendue (Rebelo 2012,
Calvo-Zaragoza 2020) ; seules les approches **réseau** dépassent ~90 % sur PrIMuS.
Nos chiffres sont donc cohérents avec l'état de l'art du classique.

## 6. Glossaire / FAQ soutenance

- **Armure** (*key signature*) : les dièses (♯) / bémols (♭) placés juste après
  la clef, valables pour tout le morceau (ex. 2 ♯ = Ré majeur → tous les fa/do
  sont ♯). Sans la lire, on met de mauvaises altérations sur les notes.
- **Silence** (*rest*) : symbole de **pause** (on ne joue rien), avec une durée
  comme les notes (soupir = noire, demi-soupir = croche…).
- **Détection « auto » vs « oracle »** :
  - *auto* = le programme trouve l'info **depuis les pixels** de l'image.
  - *oracle* = on lui **donne la bonne réponse** (vérité-terrain) — c'est un
    outil de **diagnostic**, pas une méthode du pipeline, **non annoncé** dans
    l'état de l'art.
  - « **armure oracle** » encadre la réalité : la ligne **sans oracle** = notre
    programme réel (armure désactivée, aucune altération, NA 50) ; la ligne
    **oracle** = armure parfaite (NA 62). L'écart = le coût exact de l'armure
    non lue. Notre détecteur auto est trop faible (+0,3 NA) → **laissé OFF**.
- **k-NN (k plus proches voisins)** : ce **n'est pas** un réseau de neurones.
  Pour classer une forme, on regarde les k exemples connus les plus ressemblants
  et on vote. « Entraîner » = **mémoriser des exemples** (aucun poids, aucun
  gradient). Méthode MLRF, autorisée par la contrainte « sans réseau ».
- **Hough** (transformée de) : détection de **droites** par vote des pixels ;
  utile pour des portées **inclinées**. Chez nous : testée, meilleure que la
  projection sur images déformées (99 % vs 61 %). ≠ **Hu**.
- **Hu (moments de)** : 7 **descripteurs de forme** invariants (position/
  rotation/taille), utilisés dans nos classifieurs k-NN. Rien à voir avec Hough
  malgré le nom.
- **Chemins stables** (Cardoso 2009) : suppression de portées par plus court
  chemin suivant la ligne ; utile pour lignes **courbes**. Testée : équivalente
  sur imprimé, meilleure sur déformé (PA diat. 38 → 43).
- **PA diatonique vs stricte** : diatonique = hauteur juste **en ignorant** les
  altérations ; stricte = altérations comprises. L'écart entre les deux = ce que
  coûte l'armure.
