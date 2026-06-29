from omr.io_primus import parse_semantic


def test_parse_semantic_notes_and_rests(tmp_path):
    content = ("clef-C1\tkeySignature-DM\ttimeSignature-2/4\t"
               "note-A4_eighth\tnote-B4_eighth.\trest-quarter\t"
               "note-C#5_quarter_fermata\tbarline\ttie")
    f = tmp_path / "x.semantic"
    f.write_text(content, encoding="utf-8")

    notes = parse_semantic(str(f))
    assert notes == [
        {"pitch": "A4", "duration": "eighth"},
        {"pitch": "B4", "duration": "eighth."},
        {"pitch": "rest", "duration": "quarter"},
        {"pitch": "C#5", "duration": "quarter"},
    ]
