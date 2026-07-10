#include "midi.hpp"

#include <fstream>
#include <unordered_map>

#include "util.hpp"

namespace omr {

namespace {

const std::unordered_map<char, int>& semitone_table() {
    static const std::unordered_map<char, int> t = {{'C', 0}, {'D', 2}, {'E', 4}, {'F', 5},
                                                      {'G', 7}, {'A', 9}, {'B', 11}};
    return t;
}

const std::unordered_map<std::string, double>& quarters_table() {
    static const std::unordered_map<std::string, double> t = {
        {"whole", 4.0},     {"half", 2.0},         {"quarter", 1.0},
        {"eighth", 0.5},    {"sixteenth", 0.25},   {"thirty_second", 0.125},
        {"sixty_fourth", 0.0625}};
    return t;
}

std::vector<uint8_t> vlq(int64_t value) {
    std::vector<uint8_t> buf = {static_cast<uint8_t>(value & 0x7F)};
    value >>= 7;
    while (value) {
        buf.insert(buf.begin(), static_cast<uint8_t>((value & 0x7F) | 0x80));
        value >>= 7;
    }
    return buf;
}

void append(std::vector<uint8_t>& out, const std::vector<uint8_t>& bytes) {
    out.insert(out.end(), bytes.begin(), bytes.end());
}

void append_u32be(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>((v >> 24) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 16) & 0xFF));
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

void append_u16be(std::vector<uint8_t>& out, uint16_t v) {
    out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
    out.push_back(static_cast<uint8_t>(v & 0xFF));
}

}  // namespace

int pitch_to_midi(const std::string& pitch) {
    char letter = pitch[0];
    size_t i = 1;
    int acc = 0;
    while (i < pitch.size() && (pitch[i] == '#' || pitch[i] == 'b')) {
        acc += (pitch[i] == '#') ? 1 : -1;
        ++i;
    }
    int octave = std::stoi(pitch.substr(i));
    return 12 * (octave + 1) + semitone_table().at(letter) + acc;
}

int duration_to_ticks(const std::string& duration, int tpq) {
    std::string base = duration;
    int dots = 0;
    while (!base.empty() && base.back() == '.') {
        ++dots;
        base.pop_back();
    }
    auto it = quarters_table().find(base);
    double q = it != quarters_table().end() ? it->second : 1.0;
    double total = q, add = q;
    for (int i = 0; i < dots; ++i) {
        add /= 2;
        total += add;
    }
    return static_cast<int>(py_round(total * tpq));
}

void write_midi(const std::vector<NoteEntry>& notes, const std::string& path, int tpq,
                 int velocity) {
    std::vector<uint8_t> track;
    int64_t pending = 0;
    for (const NoteEntry& n : notes) {
        int ticks = duration_to_ticks(n.duration, tpq);
        if (n.pitch == "rest") {
            pending += ticks;
            continue;
        }
        int key = pitch_to_midi(n.pitch);
        append(track, vlq(pending));
        track.push_back(0x90);
        track.push_back(static_cast<uint8_t>(key));
        track.push_back(static_cast<uint8_t>(velocity));
        append(track, vlq(ticks));
        track.push_back(0x80);
        track.push_back(static_cast<uint8_t>(key));
        track.push_back(0x00);
        pending = 0;
    }
    append(track, vlq(pending));
    track.push_back(0xFF);
    track.push_back(0x2F);
    track.push_back(0x00);

    std::vector<uint8_t> header_data;
    append_u32be(header_data, 6);
    append_u16be(header_data, 0);
    append_u16be(header_data, 1);
    append_u16be(header_data, static_cast<uint16_t>(tpq));

    std::vector<uint8_t> track_len;
    append_u32be(track_len, static_cast<uint32_t>(track.size()));

    std::ofstream f(path, std::ios::binary);
    f.write("MThd", 4);
    f.write(reinterpret_cast<const char*>(header_data.data()),
            static_cast<std::streamsize>(header_data.size()));
    f.write("MTrk", 4);
    f.write(reinterpret_cast<const char*>(track_len.data()),
            static_cast<std::streamsize>(track_len.size()));
    f.write(reinterpret_cast<const char*>(track.data()), static_cast<std::streamsize>(track.size()));
}

}  // namespace omr
