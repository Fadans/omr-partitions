#include "knn.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace omr {

void NearestNeighbor::fit(const std::vector<std::vector<double>>& X,
                           const std::vector<std::string>& y) {
    size_t n = X.size();
    size_t d = n ? X[0].size() : 0;
    mean_.assign(d, 0.0);
    std_.assign(d, 0.0);
    for (const auto& row : X) {
        for (size_t j = 0; j < d; ++j) mean_[j] += row[j];
    }
    for (size_t j = 0; j < d; ++j) mean_[j] /= static_cast<double>(n);
    for (const auto& row : X) {
        for (size_t j = 0; j < d; ++j) {
            double diff = row[j] - mean_[j];
            std_[j] += diff * diff;
        }
    }
    for (size_t j = 0; j < d; ++j) {
        std_[j] = std::sqrt(std_[j] / static_cast<double>(n));
        if (std_[j] == 0.0) std_[j] = 1.0;
    }
    X_.assign(n, std::vector<double>(d));
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < d; ++j) X_[i][j] = (X[i][j] - mean_[j]) / std_[j];
    }
    y_ = y;
}

std::string NearestNeighbor::predict_one(const std::vector<double>& v) const {
    size_t d = v.size();
    std::vector<double> vs(d);
    for (size_t j = 0; j < d; ++j) vs[j] = (v[j] - mean_[j]) / std_[j];

    size_t n = X_.size();
    std::vector<double> dist(n);
    for (size_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < d; ++j) {
            double diff = X_[i][j] - vs[j];
            s += diff * diff;
        }
        dist[i] = std::sqrt(s);
    }

    std::vector<size_t> idx(n);
    for (size_t i = 0; i < n; ++i) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(),
                      [&](size_t a, size_t b) { return dist[a] < dist[b]; });

    int kk = std::min<int>(k_, static_cast<int>(n));
    std::unordered_map<std::string, int> counts;
    std::vector<std::string> order;
    for (int i = 0; i < kk; ++i) {
        const std::string& lab = y_[idx[i]];
        if (counts.find(lab) == counts.end()) order.push_back(lab);
        counts[lab] += 1;
    }
    std::string best = order[0];
    int best_count = counts[best];
    for (const std::string& lab : order) {
        if (counts[lab] > best_count) {
            best = lab;
            best_count = counts[lab];
        }
    }
    return best;
}

}  // namespace omr
