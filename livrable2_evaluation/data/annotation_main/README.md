# Annotation manuelle PA / RA (note interne)

Vérité-terrain des hauteurs/durées, saisie à la main sur quelques images (imprimées et manuscrites).

Un fichier JSON par image. On copie `gabarit.json`, on renomme, et on saisit la mélodie dans l'ordre de
lecture.

- `pitch` : nom + octave, ex. `C4`, `F#5`, `Bb3`.
- `duration` : `w` ronde, `h` blanche, `q` noire, `e` croche, `s` double croche.

Pas besoin de boîtes ici (`bbox` non requis pour PA/RA). Garder les fichiers courts : quelques mesures
suffisent par image.
