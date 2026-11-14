"""Weights: the state dataclass, the pluggable interface and the phase 1 default.

A weights function has the signature

    weights(state: WeightState, options: Sequence[chord]) -> list[int]

and returns one int per option. It may use only the fields of the state.
A trained model can replace the default without any other code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Sequence

from .keys import Key

Chord = tuple[int, int, int, int]


@dataclass(frozen=True)
class WeightState:
    chorale_id: int
    phrase_index: int
    beat_index: int
    key: Key
    soprano_midi: int
    prev_chord: Chord | None
    prev_prev_bass: int | None


WeightsFn = Callable[[WeightState, Sequence[Chord]], list[int]]


@lru_cache(maxsize=None)
def chord_root_pc(chord: Chord) -> int:
    """Root pitch class of a chord, from its notes alone.

    The root is the note the others stack above in 3rds (3rd, then 5th, then
    7th). A triad reading wins over a seventh-chord reading, then the lowest
    pitch class wins. A chord that is not built in 3rds gets its bass note.
    """
    pcs = sorted({m % 12 for m in chord})
    triads = []
    sevenths = []
    for p in pcs:
        above = [(q - p) % 12 for q in pcs if q != p]
        thirds = [i for i in above if i in (3, 4)]
        fifths = [i for i in above if i in (6, 7)]
        sevs = [i for i in above if i in (9, 10, 11)]
        if len(thirds) != 1 or len(fifths) > 1 or len(sevs) > 1:
            continue
        if len(thirds) + len(fifths) + len(sevs) != len(above):
            continue
        if not sevs:
            triads.append(p)
            continue
        # A diminished 7th (9 semitones) only over a diminished chord.
        if sevs[0] == 9 and not (thirds[0] == 3 and (not fifths or fifths[0] == 6)):
            continue
        sevenths.append(p)
    if triads:
        return triads[0]
    if sevenths:
        return sevenths[0]
    return chord[3] % 12


def pair_weight(prev: Chord, chord: Chord) -> int:
    """Phase 1 weight of moving from prev to chord."""
    held = 0
    moved = 0
    for v in (1, 2, 3):
        d = chord[v] - prev[v]
        if d == 0:
            held += 1
        else:
            moved += abs(d)
    root_move = (chord_root_pc(chord) - chord_root_pc(prev)) % 12
    w = 3 * held - moved
    if root_move in (5, 7):
        w += 4
    if root_move == 0 and chord[3] == prev[3]:
        w -= 6
    return w


def default_weights(state: WeightState, options: Sequence[Chord]) -> list[int]:
    """Phase 1 default. Uses only state.prev_chord.

    weight = 3 x (A/T/B voices holding the same pitch)
             - (total semitone movement of A, T, B)
             + 4 if the root moves by a perfect 4th or 5th
             - 6 if the root and the bass pitch are both unchanged.
    With no previous chord every weight is 0.
    """
    prev = state.prev_chord
    if prev is None:
        return [0] * len(options)
    prev = tuple(prev)
    return [pair_weight(prev, tuple(c)) for c in options]


def checked_weights(weights: WeightsFn, state: WeightState, options: Sequence[Chord]) -> list[int]:
    """Call a weights function and insist on one Python int per option."""
    out = list(weights(state, options))
    if len(out) != len(options):
        raise ValueError(f"weights returned {len(out)} values for {len(options)} options")
    for w in out:
        if type(w) is not int:
            raise TypeError(f"weights must be Python ints, got {type(w).__name__}")
    return out
