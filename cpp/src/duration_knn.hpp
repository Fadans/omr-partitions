#pragma once

#include <optional>
#include <string>
#include <vector>

#include "image.hpp"
#include "knn.hpp"
#include "segment.hpp"
#include "staff.hpp"

namespace omr {

std::string base_duration(const std::string& d);

std::vector<double> region_descriptor(const Mask& nostaff, const Mask& bmask,
                                       const Component& head, const StaffMetrics& metrics);

class DurationClassifier {
public:
    explicit DurationClassifier(int k = 5) : knn(k) {}

    std::string predict_head(const Mask& nostaff, const Mask& bmask, const Component& head,
                              const StaffMetrics& metrics) const;

    static DurationClassifier load(const std::string& path);

    NearestNeighbor knn;
};

// Charge models/duration_knn.npz si présent, sinon nullopt (comme
// duration_knn.load_default en Python).
std::optional<DurationClassifier> load_default(const std::string& path);

}  // namespace omr
