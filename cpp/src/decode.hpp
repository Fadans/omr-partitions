#pragma once

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "image.hpp"
#include "segment.hpp"
#include "staff.hpp"

namespace omr {

struct NoteEntry {
    std::string pitch;
    std::string duration;
};

struct Stem {
    int x, top, bottom;
};

// Signature identique à DurationClassifier::predict_head : (nostaff, ligatures, tête, métriques).
using DurationFn =
    std::function<std::string(const Mask&, const Mask&, const Component&, const StaffMetrics&)>;

std::pair<int, int> clef_reference(const std::string& clef_token);
std::pair<double, double> staff_geometry(const Mask& mask, const StaffMetrics& metrics);
std::string head_pitch(const Component& head, double bottom_y, double half, int v_ref, int d_ref);

std::optional<Stem> find_stem(const Mask& nostaff, const Component& head,
                               const StaffMetrics& metrics);
Mask beam_mask(const Mask& nostaff, const StaffMetrics& metrics);
int count_beams(const Mask& beams_mask, const Component& head, const Stem& stem,
                 const StaffMetrics& metrics);
std::string head_duration(const Mask& nostaff, const Component& head, const StaffMetrics& metrics,
                           const Mask* beams_mask = nullptr);
std::vector<Component> clean_heads(const std::vector<Component>& heads);

std::vector<NoteEntry> decode(const Mask& mask, const Mask& nostaff, const StaffMetrics& metrics,
                               const std::string& clef_token,
                               const std::vector<Component>* heads = nullptr,
                               const DurationFn& dur_clf = nullptr);

}  // namespace omr
