#include "decode.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>

#include "heads.hpp"
#include "morph.hpp"
#include "util.hpp"

namespace omr {

namespace {

const std::string LETTERS = "CDEFGAB";

struct ClefRef {
    char letter;
    int octave;
};

const std::unordered_map<char, ClefRef>& clef_ref_table() {
    static const std::unordered_map<char, ClefRef> table = {
        {'G', {'G', 4}}, {'C', {'C', 4}}, {'F', {'F', 3}}};
    return table;
}

int pitch_value(char letter, int octave) {
    return 7 * octave + static_cast<int>(LETTERS.find(letter));
}

std::string value_to_pitch(int v) {
    char letter = LETTERS[py_mod(v, 7)];
    int octave = py_floordiv(v, 7);
    return std::string(1, letter) + std::to_string(octave);
}

std::vector<Run> column_runs_slice(const Mask& mask, int x, int y0, int y1) {
    std::vector<Run> runs;
    int n = y1 - y0;
    if (n <= 0) return runs;
    bool cur = mask.at(y0, x);
    int start = 0;
    for (int i = 1; i < n; ++i) {
        bool v = mask.at(y0 + i, x);
        if (v != cur) {
            runs.push_back({start, i - start, cur});
            start = i;
            cur = v;
        }
    }
    runs.push_back({start, n - start, cur});
    return runs;
}

const std::unordered_map<int, std::string>& beam_duration_table() {
    static const std::unordered_map<int, std::string> table = {
        {0, "quarter"}, {1, "eighth"}, {2, "sixteenth"}, {3, "thirty_second"}, {4, "sixty_fourth"}};
    return table;
}

}  // namespace

std::pair<int, int> clef_reference(const std::string& clef_token) {
    size_t dash = clef_token.find('-');
    std::string body = dash == std::string::npos ? clef_token : clef_token.substr(dash + 1);
    char letter = body[0];
    int line = body[1] - '0';
    auto it = clef_ref_table().find(letter);
    if (it == clef_ref_table().end()) throw std::runtime_error("clef inconnue : " + clef_token);
    const ClefRef& ref = it->second;
    return {pitch_value(ref.letter, ref.octave), 2 * (line - 1)};
}

std::pair<double, double> staff_geometry(const Mask& mask, const StaffMetrics& metrics) {
    std::vector<int> rows = staff_line_rows(mask);
    if (rows.size() >= 2) {
        int mn = *std::min_element(rows.begin(), rows.end());
        int mx = *std::max_element(rows.begin(), rows.end());
        double bottom = static_cast<double>(mx);
        double half = static_cast<double>(mx - mn) / (static_cast<double>(rows.size() - 1) * 2.0);
        return {bottom, half};
    }
    return {static_cast<double>(mask.h - 1), metrics.staff_unit() / 2.0};
}

std::string head_pitch(const Component& head, double bottom_y, double half, int v_ref,
                        int d_ref) {
    double cy = (head.y1 + head.y2) / 2.0;
    long long d = py_round((bottom_y - cy) / half);
    return value_to_pitch(v_ref + static_cast<int>(d - d_ref));
}

std::optional<Stem> find_stem(const Mask& nostaff, const Component& head,
                               const StaffMetrics& metrics) {
    int w = nostaff.w;
    int x0 = std::max(0, head.x1 - metrics.line_height);
    int x1 = std::min(w, head.x2 + metrics.line_height);
    int min_len = static_cast<int>(1.5 * metrics.staff_unit());

    int best_len = 0, best_x = 0, best_s = 0, best_e = 0;
    for (int x = x0; x < x1; ++x) {
        for (const Run& r : column_runs(nostaff, x)) {
            if (r.value && r.length > best_len) {
                best_len = r.length;
                best_x = x;
                best_s = r.start;
                best_e = r.start + r.length;
            }
        }
    }
    if (best_len < min_len) return std::nullopt;
    return Stem{best_x, best_s, best_e};
}

Mask beam_mask(const Mask& nostaff, const StaffMetrics& metrics) {
    int width = std::max(3, static_cast<int>(1.1 * metrics.staff_unit()));
    return opening(nostaff, rect(1, width));
}

int count_beams(const Mask& beams_mask, const Component& head, const Stem& stem,
                 const StaffMetrics& metrics) {
    double head_cy = (head.y1 + head.y2) / 2.0;
    int u = metrics.staff_unit();
    int h = beams_mask.h, w = beams_mask.w;
    int y0, y1;
    if (stem.top < head_cy) {
        y0 = std::max(0, stem.top - metrics.line_height);
        y1 = std::min(h, stem.top + 3 * u);
    } else {
        y0 = std::max(0, stem.bottom - 3 * u);
        y1 = std::min(h, stem.bottom + metrics.line_height);
    }

    int best = 0;
    for (int x = std::max(0, stem.x - 1); x < std::min(w, stem.x + 2); ++x) {
        int runs = 0;
        for (const Run& r : column_runs_slice(beams_mask, x, y0, y1)) {
            if (r.value && r.length <= 1.4 * u) runs += 1;
        }
        best = std::max(best, runs);
    }
    return best;
}

std::string head_duration(const Mask& nostaff, const Component& head, const StaffMetrics& metrics,
                           const Mask* beams_mask_in) {
    long long area = 0;
    int ch = head.height(), cw = head.width();
    for (int y = head.y1; y < head.y2; ++y) {
        for (int x = head.x1; x < head.x2; ++x) {
            if (nostaff.at(y, x)) area += 1;
        }
    }
    long long size = static_cast<long long>(ch) * cw;
    bool hollow = static_cast<double>(area) / std::max<long long>(1, size) < 0.55;

    std::optional<Stem> stem = find_stem(nostaff, head, metrics);
    if (!stem) return hollow ? "whole" : "quarter";
    if (hollow) return "half";

    Mask local_bmask;
    const Mask* bm = beams_mask_in;
    if (!bm) {
        local_bmask = beam_mask(nostaff, metrics);
        bm = &local_bmask;
    }
    int nb = count_beams(*bm, head, *stem, metrics);
    auto it = beam_duration_table().find(nb);
    return it != beam_duration_table().end() ? it->second : "sixty_fourth";
}

std::vector<Component> clean_heads(const std::vector<Component>& heads) {
    if (heads.empty()) return {};
    std::vector<int> heights;
    heights.reserve(heads.size());
    for (const Component& h : heads) heights.push_back(h.height());
    std::vector<int> sorted_h = heights;
    std::sort(sorted_h.begin(), sorted_h.end());
    size_t n = sorted_h.size();
    double med_h = (n % 2 == 1) ? sorted_h[n / 2]
                                 : (sorted_h[n / 2 - 1] + sorted_h[n / 2]) / 2.0;

    std::vector<Component> kept;
    for (const Component& h : heads) {
        if (h.height() >= 0.6 * med_h) kept.push_back(h);
    }
    std::stable_sort(kept.begin(), kept.end(),
                      [](const Component& a, const Component& b) { return a.x1 < b.x1; });

    std::vector<Component> out;
    for (const Component& h : kept) {
        if (!out.empty()) {
            const Component& prev = out.back();
            int overlap = std::min(prev.x2, h.x2) - std::max(prev.x1, h.x1);
            if (overlap > 0.5 * std::min(prev.width(), h.width())) {
                if (h.area > prev.area) out.back() = h;
                continue;
            }
        }
        out.push_back(h);
    }
    return out;
}

std::vector<NoteEntry> decode(const Mask& mask, const Mask& nostaff, const StaffMetrics& metrics,
                               const std::string& clef_token,
                               const std::vector<Component>* heads_in, const DurationFn& dur_clf) {
    std::vector<Component> heads = heads_in ? *heads_in : detect_heads(nostaff, metrics);
    heads = clean_heads(heads);

    auto [bottom_y, half] = staff_geometry(mask, metrics);
    auto [v_ref, d_ref] = clef_reference(clef_token);
    Mask bmask = beam_mask(nostaff, metrics);

    std::stable_sort(heads.begin(), heads.end(),
                      [](const Component& a, const Component& b) { return a.x1 < b.x1; });

    std::vector<std::pair<double, NoteEntry>> items;
    items.reserve(heads.size());
    for (const Component& h : heads) {
        std::string pitch = head_pitch(h, bottom_y, half, v_ref, d_ref);
        std::string duration =
            dur_clf ? dur_clf(nostaff, bmask, h, metrics) : head_duration(nostaff, h, metrics, &bmask);
        items.push_back({(h.x1 + h.x2) / 2.0, NoteEntry{pitch, duration}});
    }
    std::stable_sort(items.begin(), items.end(),
                      [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<NoteEntry> out;
    out.reserve(items.size());
    for (auto& it : items) out.push_back(std::move(it.second));
    return out;
}

}  // namespace omr
