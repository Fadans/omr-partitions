#pragma once

#include <optional>
#include <vector>

#include "image.hpp"
#include "segment.hpp"
#include "staff.hpp"

namespace omr {

Mask fill_holes(const Mask& mask, std::optional<int> max_hole_area = std::nullopt);

std::vector<Component> detect_heads(const Mask& mask, const StaffMetrics& metrics,
                                     double radius_ratio = 0.40, double min_fill = 0.55,
                                     double aspect_min = 0.55, double aspect_max = 2.4);

}  // namespace omr
