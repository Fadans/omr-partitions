#include <cstdio>
#include <string>

#include "image.hpp"
#include "preprocess.hpp"
#include "staff.hpp"

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

    omr::StaffMetrics metrics = omr::estimate_metrics(mask);
    std::vector<int> rows = omr::staff_line_rows(mask);
    omr::Mask nostaff = omr::remove_staff_lines_combined(mask, metrics, &rows);
    long long nsum = 0;
    for (uint8_t v : nostaff.data) nsum += v;
    std::printf("line_height=%d line_space=%d n_rows=%zu nostaff_sum=%lld\n", metrics.line_height,
                metrics.line_space, rows.size(), nsum);
    return 0;
}
