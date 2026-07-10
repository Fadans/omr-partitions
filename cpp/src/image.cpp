#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "image.hpp"

#include <stdexcept>

namespace omr {

Gray load_gray(const std::string& path) {
    int w, h, channels;
    unsigned char* px = stbi_load(path.c_str(), &w, &h, &channels, 0);
    if (!px) {
        throw std::runtime_error("impossible de charger l'image : " + path);
    }

    Gray g(w, h);
    for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
        const unsigned char* p = px + i * channels;
        if (channels >= 3) {
            int r = p[0], gg = p[1], b = p[2];
            g.data[i] = static_cast<uint8_t>((r * 299 + gg * 587 + b * 114 + 500) / 1000);
        } else {
            g.data[i] = p[0];
        }
    }
    stbi_image_free(px);
    return g;
}

}  // namespace omr
