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

bool has_flag(int argc, char** argv, const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (flag == argv[i]) return true;
    }
    return false;
}

std::vector<omr::NoteEntry> decode_sample(const omr::PrimusSample& sample,
                                           const omr::DurationFn& dur_fn) {
    omr::Gray gray = omr::load_gray(sample.png());
    omr::Mask mask = omr::preprocess(gray);
    omr::StaffMetrics metrics = omr::estimate_metrics(mask);
    std::vector<int> rows = omr::staff_line_rows(mask);
    omr::Mask nostaff = omr::remove_staff_lines_combined(mask, metrics, &rows);

    std::ifstream sf(sample.semantic());
    std::string clef;
    sf >> clef;

    return omr::decode(mask, nostaff, metrics, clef, nullptr, dur_fn);
}

}  // namespace

// Reproduit le chemin "clean" de omr.cli.demo() : prétraitement Otsu,
// suppression de portée, détection des têtes, décodage (k-NN de durées si le
// modèle est disponible), export MIDI.
//
// --id <id>  : mode démo, une image -> MIDI + séquence sur stdout.
// --bench    : mode vérification, décode tous les échantillons de --corpus
//              (modèle chargé une seule fois) et imprime, par échantillon,
//              "id<TAB>pitch/duration ..." sur stdout (pour cpp/verify/compare.py).
int main(int argc, char** argv) {
    std::string corpus = arg_value(argc, argv, "--corpus", "sample_corpus");
    std::string out_dir = arg_value(argc, argv, "--out", "out");
    std::string model_path = arg_value(argc, argv, "--model", "models/duration_knn.npz");
    std::string id = arg_value(argc, argv, "--id", "");
    std::string n_str = arg_value(argc, argv, "--n", "");
    bool bench = has_flag(argc, argv, "--bench");

    std::optional<omr::DurationClassifier> model = omr::load_default(model_path);
    omr::DurationFn dur_fn = nullptr;
    if (model) {
        dur_fn = [&](const omr::Mask& ns, const omr::Mask& bm, const omr::Component& h,
                     const omr::StaffMetrics& m) { return model->predict_head(ns, bm, h, m); };
    }

    if (bench) {
        std::optional<int> limit = n_str.empty() ? std::nullopt : std::optional<int>(std::stoi(n_str));
        std::vector<omr::PrimusSample> samples = omr::list_samples(corpus, limit);
        double total_ms = 0.0;
        int n_done = 0;
        for (const omr::PrimusSample& sample : samples) {
            if (!std::filesystem::exists(sample.png())) continue;
            auto t0 = std::chrono::steady_clock::now();
            std::vector<omr::NoteEntry> notes = decode_sample(sample, dur_fn);
            auto t1 = std::chrono::steady_clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ++n_done;

            std::printf("%s\t", sample.sample_id.c_str());
            for (size_t i = 0; i < notes.size(); ++i) {
                if (i) std::printf(" ");
                std::printf("%s/%s", notes[i].pitch.c_str(), notes[i].duration.c_str());
            }
            std::printf("\n");
        }
        std::fprintf(stderr, "bench: %d echantillons, %.3f ms/echantillon (moyenne)\n", n_done,
                     n_done ? total_ms / n_done : 0.0);
        return 0;
    }

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
    std::vector<omr::NoteEntry> notes = decode_sample(sample, dur_fn);
    auto t1 = std::chrono::steady_clock::now();

    std::filesystem::create_directories(out_dir);
    omr::write_midi(notes, out_dir + "/" + sample.sample_id + ".mid");

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::fprintf(stderr, "%s: %zu notes, temps=%.3fms\n", sample.sample_id.c_str(), notes.size(),
                 ms);

    for (const omr::NoteEntry& n : notes) std::printf("%s/%s\n", n.pitch.c_str(), n.duration.c_str());
    return 0;
}
