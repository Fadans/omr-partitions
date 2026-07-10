#pragma once

#include <vector>

#include "image.hpp"

namespace omr {

struct Component {
    int label;
    int x1, y1, x2, y2;
    int area;

    int width() const { return x2 - x1; }
    int height() const { return y2 - y1; }
};

// Étiquetage en composantes connexes 8-connexité (union-find). Renvoie une
// carte d'étiquettes (0 = fond, 1..n = composantes) de la même taille que mask.
std::vector<int32_t> label_components(const Mask& mask);

std::vector<Component> components_from_labels(const std::vector<int32_t>& labels, int w, int h);

// Composantes de taille >= min_area, triées par abscisse gauche (x1).
std::vector<Component> segment(const Mask& mask, int min_area = 8);

}  // namespace omr
