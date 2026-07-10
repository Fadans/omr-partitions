#pragma once

#include <optional>

#include "image.hpp"

namespace omr {

Gray median_blur(const Gray& img, int k = 3);
int otsu_threshold(const Gray& gray);
Mask binarize(const Gray& gray, std::optional<int> threshold = std::nullopt,
              bool adaptive = false, int window = 25, double c = 8.0);
Mask preprocess(const Gray& gray, int median_k = 3, bool adaptive = false);

}  // namespace omr
