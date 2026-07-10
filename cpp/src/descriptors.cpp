#include "descriptors.hpp"

#include <cmath>

namespace omr {

std::array<double, 7> hu_moments(const Mask& crop) {
    std::vector<double> xs, ys;
    for (int y = 0; y < crop.h; ++y) {
        for (int x = 0; x < crop.w; ++x) {
            if (crop.at(y, x)) {
                xs.push_back(x);
                ys.push_back(y);
            }
        }
    }
    std::array<double, 7> out{};
    if (xs.empty()) return out;

    double m00 = static_cast<double>(xs.size());
    double xb = 0.0, yb = 0.0;
    for (double v : xs) xb += v;
    for (double v : ys) yb += v;
    xb /= m00;
    yb /= m00;

    std::vector<double> dx(xs.size()), dy(ys.size());
    for (size_t i = 0; i < xs.size(); ++i) {
        dx[i] = xs[i] - xb;
        dy[i] = ys[i] - yb;
    }

    auto mu = [&](int p, int q) {
        double s = 0.0;
        for (size_t i = 0; i < dx.size(); ++i) s += std::pow(dx[i], p) * std::pow(dy[i], q);
        return s;
    };
    auto eta = [&](int p, int q) { return mu(p, q) / std::pow(m00, 1.0 + (p + q) / 2.0); };

    double n20 = eta(2, 0), n02 = eta(0, 2), n11 = eta(1, 1);
    double n30 = eta(3, 0), n03 = eta(0, 3), n21 = eta(2, 1), n12 = eta(1, 2);

    std::array<double, 7> h{};
    h[0] = n20 + n02;
    h[1] = (n20 - n02) * (n20 - n02) + 4 * n11 * n11;
    h[2] = (n30 - 3 * n12) * (n30 - 3 * n12) + (3 * n21 - n03) * (3 * n21 - n03);
    h[3] = (n30 + n12) * (n30 + n12) + (n21 + n03) * (n21 + n03);
    h[4] = (n30 - 3 * n12) * (n30 + n12) *
               ((n30 + n12) * (n30 + n12) - 3 * (n21 + n03) * (n21 + n03)) +
           (3 * n21 - n03) * (n21 + n03) *
               (3 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));
    h[5] = (n20 - n02) * ((n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03)) +
           4 * n11 * (n30 + n12) * (n21 + n03);
    h[6] = (3 * n21 - n03) * (n30 + n12) *
               ((n30 + n12) * (n30 + n12) - 3 * (n21 + n03) * (n21 + n03)) -
           (n30 - 3 * n12) * (n21 + n03) *
               (3 * (n30 + n12) * (n30 + n12) - (n21 + n03) * (n21 + n03));

    for (int i = 0; i < 7; ++i) {
        double sign = (h[i] > 0) - (h[i] < 0);
        out[i] = -sign * std::log10(std::abs(h[i]) + 1e-30);
    }
    return out;
}

namespace {

std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> out(num);
    if (num == 1) {
        out[0] = start;
        return out;
    }
    double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i) out[i] = start + step * i;
    out[num - 1] = stop;
    return out;
}

double interp_at(double x, const std::vector<double>& xp, const std::vector<double>& fp) {
    int n = static_cast<int>(xp.size());
    if (n == 1) return fp[0];
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (xp[mid] <= x) lo = mid; else hi = mid;
    }
    double t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] + t * (fp[hi] - fp[lo]);
}

std::vector<double> resample(const std::vector<double>& profile, int bins) {
    if (profile.empty()) return std::vector<double>(bins, 0.0);
    std::vector<double> xp = linspace(0.0, 1.0, static_cast<int>(profile.size()));
    std::vector<double> x = linspace(0.0, 1.0, bins);
    std::vector<double> out(bins);
    for (int i = 0; i < bins; ++i) out[i] = interp_at(x[i], xp, profile);
    double s = 0.0;
    for (double v : out) s += v;
    if (s > 0.0) {
        for (double& v : out) v /= s;
    }
    return out;
}

}  // namespace

std::vector<double> projection_features(const Mask& crop, int bins) {
    std::vector<double> row(crop.h, 0.0), col(crop.w, 0.0);
    for (int y = 0; y < crop.h; ++y) {
        for (int x = 0; x < crop.w; ++x) {
            if (crop.at(y, x)) {
                row[y] += 1.0;
                col[x] += 1.0;
            }
        }
    }
    std::vector<double> rr = resample(row, bins);
    std::vector<double> rc = resample(col, bins);
    std::vector<double> out;
    out.reserve(rr.size() + rc.size());
    out.insert(out.end(), rr.begin(), rr.end());
    out.insert(out.end(), rc.begin(), rc.end());
    return out;
}

Mask crop_of(const Mask& mask, const Component& comp) {
    int w = comp.width(), h = comp.height();
    Mask out(w, h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            out.set(y, x, mask.at(comp.y1 + y, comp.x1 + x));
        }
    }
    return out;
}

std::vector<double> shape_features(const Mask& crop, int bins) {
    int h = crop.h, w = crop.w;
    double aspect = h ? static_cast<double>(w) / h : 0.0;
    long long area = 0;
    for (uint8_t v : crop.data) area += v;
    double fill = (h && w) ? static_cast<double>(area) / (static_cast<double>(h) * w) : 0.0;

    std::array<double, 7> hu = hu_moments(crop);
    std::vector<double> pf = projection_features(crop, bins);

    std::vector<double> out;
    out.reserve(7 + pf.size() + 2);
    out.insert(out.end(), hu.begin(), hu.end());
    out.insert(out.end(), pf.begin(), pf.end());
    out.push_back(aspect);
    out.push_back(fill);
    return out;
}

std::vector<double> describe(const Mask& mask, const Component& comp, int bins) {
    return shape_features(crop_of(mask, comp), bins);
}

}  // namespace omr
