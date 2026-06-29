from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

_DURATIONS = {
    "whole", "half", "quarter", "eighth", "sixteenth",
    "thirty_second", "sixty_fourth", "hundred_twenty_eighth",
}


def _norm_duration(token: str) -> str:
    token = token.replace("_fermata", "")
    dots = ""
    while token.endswith("."):
        dots += "."
        token = token[:-1]
    return token + dots if token in _DURATIONS else token + dots


@dataclass
class PrimusSample:
    sample_id: str
    directory: str

    @property
    def png(self) -> str:
        return os.path.join(self.directory, self.sample_id + ".png")

    @property
    def distorted(self) -> str:
        return os.path.join(self.directory, self.sample_id + "_distorted.jpg")

    @property
    def semantic(self) -> str:
        return os.path.join(self.directory, self.sample_id + ".semantic")

    notes: list = field(default_factory=list)


def parse_semantic(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        tokens = f.read().split()

    notes: list[dict] = []
    for tok in tokens:
        kind, _, value = tok.partition("-")
        if kind == "note":
            pitch, _, dur = value.partition("_")
            notes.append({"pitch": pitch, "duration": _norm_duration(dur)})
        elif kind == "rest":
            notes.append({"pitch": "rest", "duration": _norm_duration(value)})
    return notes


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8)


def list_samples(corpus_dir: str, limit: int | None = None) -> list[PrimusSample]:
    samples: list[PrimusSample] = []
    for name in sorted(os.listdir(corpus_dir)):
        directory = os.path.join(corpus_dir, name)
        if not os.path.isdir(directory):
            continue
        sample = PrimusSample(sample_id=name, directory=directory)
        if not os.path.exists(sample.semantic):
            continue
        samples.append(sample)
        if limit is not None and len(samples) >= limit:
            break
    return samples
