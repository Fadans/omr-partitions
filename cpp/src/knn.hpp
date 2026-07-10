#pragma once

#include <string>
#include <vector>

namespace omr {

class NearestNeighbor {
public:
    explicit NearestNeighbor(int k = 1) : k_(k) {}

    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::string>& y);
    std::string predict_one(const std::vector<double>& v) const;

    // Accès direct pour le chargement d'un modèle déjà entraîné (npz).
    int k_value() const { return k_; }
    void set_k(int k) { k_ = k; }
    std::vector<std::vector<double>>& data() { return X_; }
    std::vector<std::string>& labels() { return y_; }
    std::vector<double>& mean_vec() { return mean_; }
    std::vector<double>& std_vec() { return std_; }

private:
    int k_;
    std::vector<std::vector<double>> X_;
    std::vector<std::string> y_;
    std::vector<double> mean_;
    std::vector<double> std_;
};

}  // namespace omr
