#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "decode.hpp"
#include "duration_knn.hpp"
#include "heads.hpp"
#include "image.hpp"
#include "io_primus.hpp"
#include "midi.hpp"
#include "preprocess.hpp"
#include "segment.hpp"
#include "staff.hpp"

namespace {

std::string arg_value(int argc, char** argv, const std::string& flag, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (flag == argv[i]) return argv[i + 1];
    }
    return def;
}

}  // namespace

// Reproduit le chemin "clean" de omr.cli.demo() : prétraitement Otsu,
// suppression de portée, détection des têtes, décodage (k-NN de durées si le
// modèle est disponible), export MIDI. La séquence décodée est imprimée sur
// stdout (pitch/duration, une par ligne) pour le harnais de comparaison.
int main(int argc, char** argv) {
    std::string corpus = arg_value(argc, argv, "--corpus", "sample_corpus");
    std::string out_dir = arg_value(argc, argv, "--out", "out");
    std::string model_path = arg_value(argc, argv, "--model", "models/duration_knn.npz");
    std::string id = arg_value(argc, argv, "--id", "");

    omr::PrimusSample sample;
    if (!id.empty()) {
        sample = {id, corpus + "/" + id};
    } else {
        std::vector<omr::PrimusSample> samples = omr::list_samples(corpus, 1);
        if (samples.empty()) {
            std::fprintf(stderr, "aucun echantillon dans %s\n", corpus.c_str());
            return 1;
        }
        sample = samples[0];
    }

    auto t0 = std::chrono::steady_clock::now();

    omr::Gray gray = omr::load_gray(sample.png());
    omr::Mask mask = omr::preprocess(gray);
    omr::StaffMetrics metrics = omr::estimate_metrics(mask);
    std::vector<int> rows = omr::staff_line_rows(mask);
    omr::Mask nostaff = omr::remove_staff_lines_combined(mask, metrics, &rows);
    std::vector<omr::Component> comps =
        omr::segment(nostaff, metrics.line_height * metrics.line_space);
    std::vector<omr::Component> heads = omr::detect_heads(nostaff, metrics);

    std::ifstream sf(sample.semantic());
    std::string clef;
    sf >> clef;

    std::optional<omr::DurationClassifier> model = omr::load_default(model_path);
    omr::DurationFn dur_fn = nullptr;
    if (model) {
        dur_fn = [&](const omr::Mask& ns, const omr::Mask& bm, const omr::Component& h,
                     const omr::StaffMetrics& m) { return model->predict_head(ns, bm, h, m); };
    }

    std::vector<omr::NoteEntry> notes = omr::decode(mask, nostaff, metrics, clef, nullptr, dur_fn);

    auto t1 = std::chrono::steady_clock::now();

    std::filesystem::create_directories(out_dir);
    omr::write_midi(notes, out_dir + "/" + sample.sample_id + ".mid");

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr,
                 "%s: interligne=%d epaisseur=%d lignes=%zu symboles=%zu tetes=%zu temps=%.3fms\n",
                 sample.sample_id.c_str(), metrics.line_space, metrics.line_height, rows.size(),
                 comps.size(), heads.size(), ms);

    for (const omr::NoteEntry& n : notes) std::printf("%s/%s\n", n.pitch.c_str(), n.duration.c_str());
    return 0;
}
