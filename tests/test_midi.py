import struct

from omr.midi import duration_to_ticks, pitch_to_midi, write_midi


def test_pitch_to_midi():
    assert pitch_to_midi("C4") == 60
    assert pitch_to_midi("A4") == 69
    assert pitch_to_midi("F#4") == 66
    assert pitch_to_midi("Bb3") == 58


def test_duration_to_ticks():
    assert duration_to_ticks("quarter", 480) == 480
    assert duration_to_ticks("eighth", 480) == 240
    assert duration_to_ticks("half", 480) == 960
    assert duration_to_ticks("quarter.", 480) == 720


def test_write_midi_header_and_readable(tmp_path):
    p = tmp_path / "x.mid"
    write_midi([{"pitch": "C4", "duration": "quarter"},
                {"pitch": "rest", "duration": "eighth"},
                {"pitch": "E4", "duration": "half"}], str(p))
    data = p.read_bytes()
    assert data[:4] == b"MThd"
    length, fmt, ntrk, tpq = struct.unpack(">IHHH", data[4:14])
    assert (length, fmt, ntrk, tpq) == (6, 0, 1, 480)
    assert b"MTrk" in data
    assert data[-3:] == bytes([0xFF, 0x2F, 0x00])
