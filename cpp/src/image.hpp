#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace omr {

// Image en niveaux de gris, indexation (y, x) comme un tableau NumPy.
struct Gray {
    int w = 0, h = 0;
    std::vector<uint8_t> data;

    Gray() = default;
    Gray(int w_, int h_) : w(w_), h(h_), data(static_cast<size_t>(w_) * h_) {}

    uint8_t at(int y, int x) const { return data[static_cast<size_t>(y) * w + x]; }
    uint8_t& at(int y, int x) { return data[static_cast<size_t>(y) * w + x]; }
};

// Masque binaire (0/1), même convention d'indexation.
struct Mask {
    int w = 0, h = 0;
    std::vector<uint8_t> data;

    Mask() = default;
    Mask(int w_, int h_, uint8_t fill = 0)
        : w(w_), h(h_), data(static_cast<size_t>(w_) * h_, fill) {}

    bool at(int y, int x) const { return data[static_cast<size_t>(y) * w + x] != 0; }
    void set(int y, int x, bool v) { data[static_cast<size_t>(y) * w + x] = v ? 1 : 0; }
};

// Charge une image (PNG/JPG) et la convertit en niveaux de gris avec la
// formule de PIL ("L" = (R*299 + G*587 + B*114 + 500) / 1000).
Gray load_gray(const std::string& path);

}  // namespace omr
