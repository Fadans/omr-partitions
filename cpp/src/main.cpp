#include <cstdio>
#include <string>

#include "image.hpp"
#include "preprocess.hpp"

// Squelette temporaire : chaque module ajoute son étape au fur et à mesure
// des commits. Complété en exécutable final dans le module io_primus/main.
int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <image.png>\n", argv[0]);
        return 1;
    }
    omr::Gray gray = omr::load_gray(argv[1]);
    int t = omr::otsu_threshold(gray);
    omr::Mask mask = omr::preprocess(gray);
    long long sum = 0;
    for (uint8_t v : mask.data) sum += v;
    std::printf("otsu=%d mask_sum=%lld w=%d h=%d\n", t, sum, gray.w, gray.h);
    return 0;
}
