#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace omr {

// Un tableau NumPy chargé depuis un .npy : type, forme et octets bruts (ordre
// C, little-endian). Seuls les dtypes float64 ('<f8'), int64 ('<i8') et
// chaîne unicode ('<Un') sont utilisés par le modèle de durées.
struct NpyArray {
    std::string descr;
    std::vector<size_t> shape;
    std::vector<uint8_t> raw;

    std::vector<double> as_f64() const;
    std::vector<int64_t> as_i64() const;
    std::vector<std::string> as_unicode() const;
};

// Lit un .npz produit par np.savez (ZIP non compressé, ZIP_STORED) : renvoie
// les tableaux indexés par leur nom (sans l'extension .npy).
std::unordered_map<std::string, NpyArray> read_npz(const std::string& path);

}  // namespace omr
