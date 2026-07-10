#include "staff.hpp"

#include <algorithm>

#include "morph.hpp"

namespace omr {

std::vector<Run> column_runs(const Mask& mask, int x) {
    std::vector<Run> runs;
    int h = mask.h;
    if (h == 0) return runs;
    bool cur = mask.at(0, x);
    int start = 0;
    for (int y = 1; y < h; ++y) {
        bool v = mask.at(y, x);
        if (v != cur) {
            runs.push_back({start, y - start, cur});
            start = y;
            cur = v;
        }
    }
    runs.push_back({start, h - start, cur});
    return runs;
}

StaffMetrics estimate_metrics(const Mask& mask) {
    std::vector<int64_t> black(mask.h + 1, 0), white(mask.h + 1, 0);
    for (int x = 0; x < mask.w; ++x) {
        for (const Run& r : column_runs(mask, x)) {
            auto& arr = r.value ? black : white;
            arr[r.length] += 1;
        }
    }
    auto argmax_from1 = [&](const std::vector<int64_t>& arr) {
        int best_i = 1;
        int64_t best_v = arr[1];
        for (int i = 2; i <= mask.h; ++i) {
            if (arr[i] > best_v) {
                best_v = arr[i];
                best_i = i;
            }
        }
        return best_i;
    };
    return {argmax_from1(black), argmax_from1(white)};
}

std::vector<int> staff_line_rows(const Mask& mask, double ratio) {
    std::vector<int64_t> proj(mask.h, 0);
    int64_t max_proj = 0;
    for (int y = 0; y < mask.h; ++y) {
        int64_t s = 0;
        for (int x = 0; x < mask.w; ++x) s += mask.at(y, x);
        proj[y] = s;
        max_proj = std::max(max_proj, s);
    }
    std::vector<int> rows;
    if (max_proj == 0) return rows;
    double threshold = ratio * static_cast<double>(max_proj);
    int start = -1;
    for (int y = 0; y < mask.h; ++y) {
        bool on = static_cast<double>(proj[y]) >= threshold;
        if (on && start < 0) {
            start = y;
        } else if (!on && start >= 0) {
            rows.push_back((start + y - 1) / 2);
            start = -1;
        }
    }
    if (start >= 0) rows.push_back((start + mask.h - 1) / 2);
    return rows;
}

Mask remove_staff_lines(const Mask& mask, const StaffMetrics& metrics) {
    int height = 2 * metrics.line_height + 1;
    return opening(mask, rect(height, 1));
}

Mask remove_staff_lines_bridge(const Mask& mask, const StaffMetrics& metrics,
                                const std::vector<int>* rows_ptr) {
    std::vector<int> owned;
    if (!rows_ptr) {
        owned = staff_line_rows(mask);
        rows_ptr = &owned;
    }
    const std::vector<int>& rows = *rows_ptr;
    if (rows.size() < 2) return remove_staff_lines(mask, metrics);

    Mask out = mask;
    int th = metrics.line_height;
    int gap = 2;
    int h = mask.h, w = mask.w;
    for (int r : rows) {
        int r0 = std::max(0, r - th), r1 = std::min(h, r + th + 1);
        int a0 = std::max(0, r0 - gap), b1 = std::min(h, r1 + gap);
        for (int x = 0; x < w; ++x) {
            bool band_any = false;
            for (int y = r0; y < r1; ++y) {
                if (mask.at(y, x)) {
                    band_any = true;
                    break;
                }
            }
            if (!band_any) continue;
            bool above_any = false;
            if (r0 > a0) {
                for (int y = a0; y < r0; ++y) {
                    if (mask.at(y, x)) {
                        above_any = true;
                        break;
                    }
                }
            }
            bool below_any = false;
            if (b1 > r1) {
                for (int y = r1; y < b1; ++y) {
                    if (mask.at(y, x)) {
                        below_any = true;
                        break;
                    }
                }
            }
            bool bridged = above_any && below_any;
            if (!bridged) {
                for (int y = r0; y < r1; ++y) out.set(y, x, false);
            }
        }
    }
    return out;
}

Mask remove_staff_lines_combined(const Mask& mask, const StaffMetrics& metrics,
                                  const std::vector<int>* rows_ptr) {
    Mask a = remove_staff_lines(mask, metrics);
    Mask b = remove_staff_lines_bridge(mask, metrics, rows_ptr);
    Mask out(mask.w, mask.h);
    for (size_t i = 0; i < out.data.size(); ++i) out.data[i] = (a.data[i] || b.data[i]) ? 1 : 0;
    return out;
}

}  // namespace omr
