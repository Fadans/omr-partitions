from __future__ import annotations

import struct

_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_QUARTERS = {
    "whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5,
    "sixteenth": 0.25, "thirty_second": 0.125, "sixty_fourth": 0.0625,
}


def pitch_to_midi(pitch: str) -> int:
    letter = pitch[0]
    i = 1
    acc = 0
    while i < len(pitch) and pitch[i] in "#b":
        acc += 1 if pitch[i] == "#" else -1
        i += 1
    octave = int(pitch[i:])
    return 12 * (octave + 1) + _SEMITONE[letter] + acc


def duration_to_ticks(duration: str, tpq: int = 480) -> int:
    base = duration
    dots = 0
    while base.endswith("."):
        dots += 1
        base = base[:-1]
    q = _QUARTERS.get(base, 1.0)
    total = q
    add = q
    for _ in range(dots):
        add /= 2
        total += add
    return int(round(total * tpq))


def _vlq(value: int) -> bytes:
    buf = bytes([value & 0x7F])
    value >>= 7
    while value:
        buf = bytes([(value & 0x7F) | 0x80]) + buf
        value >>= 7
    return buf


def write_midi(notes: list[dict], path: str, tpq: int = 480, velocity: int = 64) -> None:
    track = bytearray()
    pending = 0
    for n in notes:
        ticks = duration_to_ticks(n["duration"], tpq)
        if n["pitch"] == "rest":
            pending += ticks
            continue
        key = pitch_to_midi(n["pitch"])
        track += _vlq(pending) + bytes([0x90, key, velocity])
        track += _vlq(ticks) + bytes([0x80, key, 0])
        pending = 0
    track += _vlq(pending) + bytes([0xFF, 0x2F, 0x00])

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, tpq)
    chunk = b"MTrk" + struct.pack(">I", len(track)) + bytes(track)
    with open(path, "wb") as f:
        f.write(header + chunk)
