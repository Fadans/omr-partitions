#pragma once

#include "image.hpp"

namespace omr {

// Élément structurant : masque binaire, même convention que Mask.
using StructElem = Mask;

StructElem disk(int radius);
StructElem rect(int height, int width);

Mask erode(const Mask& mask, const StructElem& se);
Mask dilate(const Mask& mask, const StructElem& se);
Mask opening(const Mask& mask, const StructElem& se);
Mask closing(const Mask& mask, const StructElem& se);

}  // namespace omr
