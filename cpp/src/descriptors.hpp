#pragma once

#include <array>
#include <vector>

#include "image.hpp"
#include "segment.hpp"

namespace omr {

std::array<double, 7> hu_moments(const Mask& crop);
std::vector<double> projection_features(const Mask& crop, int bins = 8);
Mask crop_of(const Mask& mask, const Component& comp);
std::vector<double> shape_features(const Mask& crop, int bins = 8);
std::vector<double> describe(const Mask& mask, const Component& comp, int bins = 8);

}  // namespace omr
