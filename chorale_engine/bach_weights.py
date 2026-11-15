"""Experimental weights learned from Bach's own harmonizations.

A counting model, trained by scripts/train_bach_model.py on Bach's chords at
every beat of the corpus chorales. The weight of an option is the sum of
integer log-frequencies (x100) of how Bach-like it is:

  - the chord it is, given the chord before (root degree, quality, inversion);
  - how far the alto, tenor and bass each move;
  - how the bass moves against the soprano (contrary, similar, oblique);
  - how many of the three lower voices move at all;
  - which chord tones are doubled;
  - the spacing between neighbouring voices;
  - optionally, a penalty for every semitone a voice leaps beyond a step (smoother lines).

It uses only state.key, state.soprano_midi and state.prev_chord, and is a
pure function. The table is a JSON file of ints, so encoder and decoder
agree exactly. Opt in with weights=bach_weights; the phase 1 default is
unchanged, and scores made with one do not decode with the other.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .weights import WeightState, chord_root_pc

MODEL_PATH = Path(__file__).resolve().parent / "data" / "bach_model.json"
MOVE_CLIP = {1: 7, 2: 7, 3: 12}  # alto, tenor, bass: semitones
# How the voices move counts this many times over, against which chord it is.
MOVE_WEIGHT = 2
# The four options at a beat differ mostly in the bass, so its own motion counts more again.
BASS_WEIGHT = 3
# Smoothness: points lost per semitone that a voice leaps beyond a whole step (0 = Bach's own taste).
SMOOTH_BASS = 300
SMOOTH_INNER = 150
# Whether anything moves at all also counts more: three still voices is the dullest outcome.
MOVERS_WEIGHT = 3


def chord_class(chord, tonic_pc: int) -> str:
    """Root degree (semitones above the tonic), 3rd, 5th, 7th and what is in the bass."""
    root = chord_root_pc(tuple(chord))
    above = {(m - root) % 12 for m in chord}
    third = 4 if 4 in above else 3 if 3 in above else 0
    fifth = 6 if 6 in above and 7 not in above else 7 if 7 in above else 0
    seventh = 1 if above & {9, 10, 11} and third else 0
    bass = (chord[3] - root) % 12
    inversion = 0 if bass == 0 else 1 if bass in (3, 4) else 2 if bass in (6, 7) else 3
    return f"{(root - tonic_pc) % 12}.{third}{fifth}{seventh}.{inversion}"


def voicing(chord) -> str:
    """Which chord tones the four voices sing, as semitones above the root."""
    root = chord_root_pc(tuple(chord))
    return ",".join(str(i) for i in sorted((m - root) % 12 for m in chord))


def clip(value: int, limit: int) -> int:
    return max(-limit, min(limit, value))


def sign(value: int) -> int:
    return (value > 0) - (value < 0)


def static_features(chord, tonic_pc: int) -> list[tuple[str, str]]:
    cls = chord_class(chord, tonic_pc)
    return [
        ("voicing", cls.split(".", 1)[1] + "|" + voicing(chord)),
        ("gap_sa", str(min(chord[0] - chord[1], 24))),
        ("gap_at", str(min(chord[1] - chord[2], 24))),
        ("gap_tb", str(min(chord[2] - chord[3], 28))),
    ]


def move_features(prev, chord) -> list[tuple[str, str]]:
    out = [(f"move{v}", str(clip(chord[v] - prev[v], MOVE_CLIP[v]))) for v in (1, 2, 3)]
    out.append(("outer", f"{sign(chord[0] - prev[0])}|{sign(chord[3] - prev[3])}"))
    # How many of alto, tenor and bass move at all: Bach almost never leaves all three still.
    out.append(("movers", str(sum(chord[v] != prev[v] for v in (1, 2, 3)))))
    return out


@lru_cache(maxsize=1)
def load_model() -> dict:
    with open(MODEL_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _lookup(table: dict, key: str) -> int:
    return table["seen"].get(key, table["unseen"])


def bach_weights(state: WeightState, options) -> list[int]:
    """One int per option: how Bach-like the option is after state.prev_chord."""
    model = load_model()
    tonic = state.key.tonic_pc
    prev = None if state.prev_chord is None else tuple(state.prev_chord)
    chords = model["chord"]
    prev_row = chords["after"].get(chord_class(prev, tonic)) if prev is not None else None
    out = []
    for option in options:
        chord = tuple(option)
        cls = chord_class(chord, tonic)
        unigram = chords["alone"]["seen"].get(cls, chords["alone"]["unseen"])
        if prev_row is not None:
            w = prev_row.get(cls, chords["backoff"] + unigram)
        else:
            w = unigram
        for name, key in static_features(chord, tonic):
            w += _lookup(model[name], key)
        if prev is not None:
            # Smoothness on top of Bach's own taste: every semitone a voice leaps beyond a step costs points.
            for v, penalty in ((1, SMOOTH_INNER), (2, SMOOTH_INNER), (3, SMOOTH_BASS)):
                w -= penalty * max(0, abs(chord[v] - prev[v]) - 2)
            for name, key in move_features(prev, chord):
                extra = BASS_WEIGHT if name == "move3" else MOVERS_WEIGHT if name == "movers" else 1
                w += extra * MOVE_WEIGHT * _lookup(model[name], key)
        out.append(w)
    return out


# The option lists do not depend on the bass two beats back, so states that
# differ only in that can share them (see options.py and codec.py).
bach_weights.ignores_prev_prev_bass = True


def with_margin(margin: int):
    """bach_weights that only offers options within `margin` of the best one at each beat.

    Fewer options means fewer bits, but the number being encoded no longer lands on a poor
    fourth choice. Every margin is its own encoding: the decoder must use the same one.
    """
    def weights(state, options):
        return bach_weights(state, options)

    weights.margin = int(margin)
    weights.ignores_prev_prev_bass = True
    weights.__name__ = f"bach_weights_margin_{int(margin)}"
    return weights
