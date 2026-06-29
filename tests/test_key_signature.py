from omr.key_signature import (
    alteration_map,
    apply_alteration,
    key_to_accidentals,
)


def test_key_to_accidentals():
    assert key_to_accidentals("keySignature-DM") == ("#", 2)
    assert key_to_accidentals("EbM") == ("b", 3)
    assert key_to_accidentals("CM") == (None, 0)


def test_alteration_map_order():
    assert alteration_map("#", 2) == {"F": "#", "C": "#"}
    assert alteration_map("b", 3) == {"B": "b", "E": "b", "A": "b"}
    assert alteration_map(None, 0) == {}


def test_apply_alteration():
    alt = alteration_map("#", 2)
    assert apply_alteration("F4", alt) == "F#4"
    assert apply_alteration("C5", alt) == "C#5"
    assert apply_alteration("G4", alt) == "G4"
    assert apply_alteration("rest", alt) == "rest"
    assert apply_alteration("F#4", alt) == "F#4"
