#include "heads.hpp"

#include <algorithm>
#include <unordered_set>

#include "morph.hpp"
#include "util.hpp"

namespace omr {

Mask fill_holes(const Mask& mask, std::optional<int> max_hole_area) {
    int h = mask.h, w = mask.w;
    Mask free(w, h);
    for (size_t i = 0; i < free.data.size(); ++i) free.data[i] = mask.data[i] ? 0 : 1;

    std::vector<int32_t> labels = label_components(free);

    std::unordered_set<int32_t> border;
    for (int x = 0; x < w; ++x) {
        border.insert(labels[x]);
        border.insert(labels[static_cast<size_t>(h - 1) * w + x]);
    }
    for (int y = 0; y < h; ++y) {
        border.insert(labels[static_cast<size_t>(y) * w]);
        border.insert(labels[static_cast<size_t>(y) * w + (w - 1)]);
    }
    border.erase(0);

    Mask out = mask;
    if (!max_hole_area) {
        for (size_t i = 0; i < out.data.size(); ++i) {
            if (free.data[i] && !border.count(labels[i])) out.data[i] = 1;
        }
        return out;
    }

    int32_t n = 0;
    for (int32_t v : labels) n = std::max(n, v);
    std::vector<int> area(n + 1, 0);
    for (int32_t v : labels) {
        if (v) area[v] += 1;
    }
    for (size_t i = 0; i < out.data.size(); ++i) {
        int32_t lab = labels[i];
        if (lab == 0 || border.count(lab)) continue;
        if (area[lab] <= *max_hole_area) out.data[i] = 1;
    }
    return out;
}

std::vector<Component> detect_heads(const Mask& mask, const StaffMetrics& metrics,
                                     double radius_ratio, double min_fill, double aspect_min,
                                     double aspect_max) {
    int radius = static_cast<int>(
        std::max<long long>(1, py_round(metrics.line_space * radius_ratio)));
    int max_hole_area = 2 * metrics.line_space * metrics.line_space;
    Mask solid = fill_holes(mask, max_hole_area);
    Mask blobs = opening(solid, disk(radius));
    std::vector<int32_t> labels = label_components(blobs);
    std::vector<Component> comps = components_from_labels(labels, blobs.w, blobs.h);

    double min_dim = 0.5 * metrics.line_space;
    double max_dim = 2.2 * metrics.line_space;

    std::vector<Component> heads;
    for (const Component& c : comps) {
        double w = c.width(), h = c.height();
        if (!(min_dim <= w && w <= max_dim && min_dim <= h && h <= max_dim)) continue;
        double aspect = w / h;
        if (!(aspect_min <= aspect && aspect <= aspect_max)) continue;
        double fill = static_cast<double>(c.area) / (w * h);
        if (fill < min_fill) continue;
        heads.push_back(c);
    }
    std::stable_sort(heads.begin(), heads.end(),
                      [](const Component& a, const Component& b) { return a.x1 < b.x1; });
    return heads;
}

}  // namespace omr
