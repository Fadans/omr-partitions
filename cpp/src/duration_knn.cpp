#include "duration_knn.hpp"

#include <algorithm>
#include <fstream>

#include "decode.hpp"
#include "descriptors.hpp"
#include "npy.hpp"

namespace omr {

std::string base_duration(const std::string& d) {
    size_t p = d.find('.');
    return p == std::string::npos ? d : d.substr(0, p);
}

namespace {

Mask extract_submask(const Mask& mask, int x0, int y0, int x1, int y1) {
    int w = std::max(0, x1 - x0), h = std::max(0, y1 - y0);
    Mask out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) out.set(y, x, mask.at(y0 + y, x0 + x));
    }
    return out;
}

}  // namespace

std::vector<double> region_descriptor(const Mask& nostaff, const Mask& bmask,
                                       const Component& head, const StaffMetrics& metrics) {
    int u = metrics.staff_unit();
    int h = nostaff.h, w = nostaff.w;
    std::optional<Stem> stem = find_stem(nostaff, head, metrics);

    long long head_area = 0;
    for (int y = head.y1; y < head.y2; ++y) {
        for (int x = head.x1; x < head.x2; ++x) {
            if (nostaff.at(y, x)) head_area += 1;
        }
    }
    long long head_size = static_cast<long long>(head.height()) * head.width();
    bool hollow = static_cast<double>(head_area) / std::max<long long>(1, head_size) < 0.55;

    int x0, y0, x1, y1;
    double stem_len;
    int nb;
    if (stem) {
        x0 = std::max(0, static_cast<int>(stem->x - 1.6 * u));
        x1 = std::min(w, static_cast<int>(stem->x + 1.6 * u));
        y0 = std::max(0, static_cast<int>(stem->top - 0.3 * u));
        y1 = std::min(h, static_cast<int>(stem->bottom + 0.3 * u));
        stem_len = static_cast<double>(stem->bottom - stem->top) / u;
        nb = count_beams(bmask, head, *stem, metrics);
    } else {
        x0 = head.x1;
        y0 = head.y1;
        x1 = head.x2;
        y1 = head.y2;
        stem_len = 0.0;
        nb = 0;
    }

    Mask crop = extract_submask(nostaff, x0, y0, x1, y1);
    int ch = std::max(1, y1 - y0), cw = std::max(1, x1 - x0);
    long long crop_area = 0;
    for (uint8_t v : crop.data) crop_area += v;
    double fill = static_cast<double>(crop_area) / (static_cast<double>(ch) * cw);
    double aspect = static_cast<double>(cw) / ch;

    std::vector<double> pf = projection_features(crop, 16);
    std::vector<double> out;
    out.reserve(pf.size() + 5);
    out.insert(out.end(), pf.begin(), pf.end());
    out.push_back(hollow ? 1.0 : 0.0);
    out.push_back(stem_len);
    out.push_back(fill);
    out.push_back(aspect);
    out.push_back(static_cast<double>(nb));
    return out;
}

std::string DurationClassifier::predict_head(const Mask& nostaff, const Mask& bmask,
                                              const Component& head,
                                              const StaffMetrics& metrics) const {
    return knn.predict_one(region_descriptor(nostaff, bmask, head, metrics));
}

DurationClassifier DurationClassifier::load(const std::string& path) {
    std::unordered_map<std::string, NpyArray> arrays = read_npz(path);

    int k = static_cast<int>(arrays.at("k").as_i64()[0]);
    DurationClassifier clf(k);

    const NpyArray& xa = arrays.at("X");
    size_t n = xa.shape.empty() ? 0 : xa.shape[0];
    size_t d = xa.shape.size() > 1 ? xa.shape[1] : 0;
    std::vector<double> flat = xa.as_f64();
    std::vector<std::vector<double>> X(n, std::vector<double>(d));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) X[i][j] = flat[i * d + j];
    }

    clf.knn.set_k(k);
    clf.knn.data() = std::move(X);
    clf.knn.mean_vec() = arrays.at("mean").as_f64();
    clf.knn.std_vec() = arrays.at("std").as_f64();
    clf.knn.labels() = arrays.at("y").as_unicode();
    return clf;
}

std::optional<DurationClassifier> load_default(const std::string& path) {
    std::ifstream test(path, std::ios::binary);
    if (!test.good()) return std::nullopt;
    return DurationClassifier::load(path);
}

}  // namespace omr
