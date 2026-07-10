#pragma once

#include <optional>
#include <string>
#include <vector>

#include "decode.hpp"

namespace omr {

struct PrimusSample {
    std::string sample_id;
    std::string directory;

    std::string png() const { return directory + "/" + sample_id + ".png"; }
    std::string distorted() const { return directory + "/" + sample_id + "_distorted.jpg"; }
    std::string semantic() const { return directory + "/" + sample_id + ".semantic"; }
};

std::vector<NoteEntry> parse_semantic(const std::string& path);
std::vector<PrimusSample> list_samples(const std::string& corpus_dir,
                                        std::optional<int> limit = std::nullopt);

}  // namespace omr
