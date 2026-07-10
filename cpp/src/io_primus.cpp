#include "io_primus.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace omr {

namespace {

std::string norm_duration(std::string token) {
    size_t p;
    while ((p = token.find("_fermata")) != std::string::npos) token.erase(p, 8);
    std::string dots;
    while (!token.empty() && token.back() == '.') {
        dots += '.';
        token.pop_back();
    }
    return token + dots;
}

}  // namespace

std::vector<NoteEntry> parse_semantic(const std::string& path) {
    std::ifstream f(path);
    std::string content((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());

    std::vector<NoteEntry> notes;
    std::istringstream iss(content);
    std::string tok;
    while (iss >> tok) {
        size_t dash = tok.find('-');
        std::string kind = dash == std::string::npos ? tok : tok.substr(0, dash);
        std::string value = dash == std::string::npos ? "" : tok.substr(dash + 1);
        if (kind == "note") {
            size_t us = value.find('_');
            std::string pitch = us == std::string::npos ? value : value.substr(0, us);
            std::string dur = us == std::string::npos ? "" : value.substr(us + 1);
            notes.push_back({pitch, norm_duration(dur)});
        } else if (kind == "rest") {
            notes.push_back({"rest", norm_duration(value)});
        }
    }
    return notes;
}

std::vector<PrimusSample> list_samples(const std::string& corpus_dir, std::optional<int> limit) {
    namespace fs = std::filesystem;
    std::vector<std::string> names;
    for (const auto& entry : fs::directory_iterator(corpus_dir)) {
        if (entry.is_directory()) names.push_back(entry.path().filename().string());
    }
    std::sort(names.begin(), names.end());

    std::vector<PrimusSample> samples;
    for (const std::string& name : names) {
        PrimusSample sample{name, corpus_dir + "/" + name};
        if (!fs::exists(sample.semantic())) continue;
        samples.push_back(sample);
        if (limit && samples.size() >= static_cast<size_t>(*limit)) break;
    }
    return samples;
}

}  // namespace omr
