#include "preprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace omr {

namespace {

int edge_index(int i, int n) { return std::clamp(i, 0, n - 1); }

// Réplique numpy.pad(mode="reflect") : réflexion sans répéter le pixel de bord.
int reflect_index(int i, int n) {
    if (n == 1) return 0;
    int period = 2 * (n - 1);
    int m = i % period;
    if (m < 0) m += period;
    return m < n ? m : period - m;
}

}  // namespace

Gray median_blur(const Gray& img, int k) {
    if (k <= 1) return img;
    int pad = k / 2;
    Gray out(img.w, img.h);
    std::vector<uint8_t> window(static_cast<size_t>(k) * k);
    for (int y = 0; y < img.h; ++y) {
        for (int x = 0; x < img.w; ++x) {
            int n = 0;
            for (int dy = 0; dy < k; ++dy) {
                int sy = edge_index(y + dy - pad, img.h);
                for (int dx = 0; dx < k; ++dx) {
                    int sx = edge_index(x + dx - pad, img.w);
                    window[n++] = img.at(sy, sx);
                }
            }
            std::sort(window.begin(), window.end());
            double median;
            if (n % 2 == 1) {
                median = window[n / 2];
            } else {
                median = (window[n / 2 - 1] + window[n / 2]) / 2.0;
            }
            out.at(y, x) = static_cast<uint8_t>(median);
        }
    }
    return out;
}

int otsu_threshold(const Gray& gray) {
    std::array<double, 256> hist{};
    for (uint8_t v : gray.data) hist[v] += 1.0;
    double total = static_cast<double>(gray.data.size());

    double sum_total = 0.0;
    for (int t = 0; t < 256; ++t) sum_total += t * hist[t];

    double w_b = 0.0, sum_b = 0.0, best_var = -1.0;
    int threshold = 0;
    for (int t = 0; t < 256; ++t) {
        w_b += hist[t];
        if (w_b == 0.0) continue;
        double w_f = total - w_b;
        if (w_f == 0.0) break;
        sum_b += t * hist[t];
        double m_b = sum_b / w_b;
        double m_f = (sum_total - sum_b) / w_f;
        double var = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if (var > best_var) {
            best_var = var;
            threshold = t;
        }
    }
    return threshold;
}

namespace {

// Moyenne locale par fenêtre glissante via image intégrale (image intégrable),
// avec un padding reflect comme numpy.pad(mode="reflect").
std::vector<double> local_mean(const Gray& gray, int window) {
    int pad = window / 2;
    int ph = gray.h + 2 * pad, pw = gray.w + 2 * pad;
    std::vector<double> p(static_cast<size_t>(ph) * pw);
    for (int y = 0; y < ph; ++y) {
        int sy = reflect_index(y - pad, gray.h);
        for (int x = 0; x < pw; ++x) {
            int sx = reflect_index(x - pad, gray.w);
            p[static_cast<size_t>(y) * pw + x] = gray.at(sy, sx);
        }
    }

    // Somme cumulée 2D avec une ligne/colonne de zéros en tête (comme le
    // np.pad constant côté haut-gauche dans le code Python).
    int csh = ph + 1, csw = pw + 1;
    std::vector<double> cs(static_cast<size_t>(csh) * csw, 0.0);
    auto cs_at = [&](int y, int x) -> double& { return cs[static_cast<size_t>(y) * csw + x]; };
    for (int y = 0; y < ph; ++y) {
        double row_sum = 0.0;
        for (int x = 0; x < pw; ++x) {
            row_sum += p[static_cast<size_t>(y) * pw + x];
            cs_at(y + 1, x + 1) = cs_at(y, x + 1) + row_sum;
        }
    }

    int s = window;
    std::vector<double> out(static_cast<size_t>(gray.w) * gray.h);
    for (int y = 0; y < gray.h; ++y) {
        for (int x = 0; x < gray.w; ++x) {
            double total = cs_at(y + s, x + s) - cs_at(y, x + s) - cs_at(y + s, x) + cs_at(y, x);
            out[static_cast<size_t>(y) * gray.w + x] = total / (static_cast<double>(s) * s);
        }
    }
    return out;
}

}  // namespace

Mask binarize(const Gray& gray, std::optional<int> threshold, bool adaptive, int window,
               double c) {
    Mask out(gray.w, gray.h);
    if (adaptive) {
        std::vector<double> lm = local_mean(gray, window);
        for (size_t i = 0; i < gray.data.size(); ++i) {
            out.data[i] = (static_cast<double>(gray.data[i]) <= lm[i] - c) ? 1 : 0;
        }
        return out;
    }
    int t = threshold ? *threshold : otsu_threshold(gray);
    for (size_t i = 0; i < gray.data.size(); ++i) {
        out.data[i] = (gray.data[i] <= t) ? 1 : 0;
    }
    return out;
}

Mask preprocess(const Gray& gray, int median_k, bool adaptive) {
    Gray smooth = median_blur(gray, median_k);
    return binarize(smooth, std::nullopt, adaptive);
}

}  // namespace omr
