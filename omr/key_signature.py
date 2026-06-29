from __future__ import annotations

SHARP_ORDER = "FCGDAEB"
FLAT_ORDER = "BEADGCF"

# tonalités majeures de PrIMuS -> (type, nombre d'altérations)
KEY_ACCIDENTALS = {
    "CM": (None, 0),
    "GM": ("#", 1), "DM": ("#", 2), "AM": ("#", 3), "EM": ("#", 4),
    "BM": ("#", 5), "F#M": ("#", 6), "C#M": ("#", 7),
    "FM": ("b", 1), "BbM": ("b", 2), "EbM": ("b", 3), "AbM": ("b", 4),
    "DbM": ("b", 5), "GbM": ("b", 6), "CbM": ("b", 7),
}


def key_to_accidentals(key_token: str) -> tuple[str | None, int]:
    body = key_token.split("-")[-1] if "-" in key_token else key_token
    return KEY_ACCIDENTALS.get(body, (None, 0))


def alteration_map(kind: str | None, count: int) -> dict[str, str]:
    if not kind or count <= 0:
        return {}
    order = SHARP_ORDER if kind == "#" else FLAT_ORDER
    return {letter: kind for letter in order[:count]}


def apply_alteration(pitch: str, alt: dict[str, str]) -> str:
    if pitch == "rest" or not alt:
        return pitch
    letter = pitch[0]
    if letter in alt and not any(c in "#b" for c in pitch[1:]):
        return letter + alt[letter] + pitch[1:]
    return pitch
