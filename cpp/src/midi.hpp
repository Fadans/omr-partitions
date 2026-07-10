#pragma once

#include <string>
#include <vector>

#include "decode.hpp"

namespace omr {

int pitch_to_midi(const std::string& pitch);
int duration_to_ticks(const std::string& duration, int tpq = 480);
void write_midi(const std::vector<NoteEntry>& notes, const std::string& path, int tpq = 480,
                 int velocity = 64);

}  // namespace omr
