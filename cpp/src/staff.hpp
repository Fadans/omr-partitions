#pragma once

#include <vector>

#include "image.hpp"

namespace omr {

struct StaffMetrics {
    int line_height;
    int line_space;
    int staff_unit() const { return line_height + line_space; }
};

struct Run {
    int start;
    int length;
    bool value;
};

// Longueurs de plages (run-length) le long de la colonne x du masque.
std::vector<Run> column_runs(const Mask& mask, int x);

StaffMetrics estimate_metrics(const Mask& mask);
std::vector<int> staff_line_rows(const Mask& mask, double ratio = 0.5);

Mask remove_staff_lines(const Mask& mask, const StaffMetrics& metrics);
Mask remove_staff_lines_bridge(const Mask& mask, const StaffMetrics& metrics,
                                const std::vector<int>* rows = nullptr);
Mask remove_staff_lines_combined(const Mask& mask, const StaffMetrics& metrics,
                                  const std::vector<int>* rows = nullptr);

}  // namespace omr
