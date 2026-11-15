"""The notes a performer plays from a written score, ornaments included.

    notes = performance(chorale_id, chords, style="fancy")
    # per voice (S, A, T, B): [(offset, duration, MIDI)], offsets and durations in quarter notes

This follows the MusicXML the engine writes: the soprano in Bach's own rhythm,
alto, tenor and bass one note per beat, and in the fancy style the walking bass
and passing notes between the beats, with trills, turns and mordents played out.
Use it for audio, so what is heard matches what is written.
"""

from __future__ import annotations

from fractions import Fraction

from .corpus import load_tune
from .musicxml_io import _segments
from .ornaments import ornament_signs, ornaments, play_sign

_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_of(pitch) -> int:
    step, alter, octave = pitch
    return 12 * (octave + 1) + _STEP_PC[step] + alter


def performance(chorale_id: int, chords, style: str | None = None,
                ornamented: bool = False, signs: bool = False) -> list[list[tuple[Fraction, Fraction, int]]]:
    """Per voice: [(offset, duration, MIDI)] as played.

    style="original" or "fancy" sets the ornaments as the score of that style has them;
    otherwise ornamented and signs choose them directly.
    """
    if style is not None:
        from .styles import style_writing

        writing = style_writing(style)
        ornamented, signs = writing["ornamented"], writing["signs"]
    tune = load_tune(chorale_id)
    chords = [tuple(int(m) for m in c) for c in chords]
    parts = _segments(tune, chords, ornaments(chorale_id, chords) if ornamented else None)
    marks = {}
    if signs:
        marks = {tune.soprano[i].offset: name for i, name in ornament_signs(chorale_id).items()}
    key_at = {beat.offset: beat.key for beat in tune.beats}
    played = []
    for v, segs in enumerate(parts):
        notes = []
        for off, length, pitch, _fermata, tie_in, _tie_out in segs:
            if pitch is None:
                continue
            midi = midi_of(pitch)
            if tie_in and notes and notes[-1][2] == midi:
                start, dur, _ = notes[-1]
                notes[-1] = (start, dur + length, midi)
                continue
            notes.append((off, length, midi))
        if v == 0 and marks:
            expanded = []
            for off, length, midi in notes:
                sign = marks.get(off)
                if sign is None:
                    expanded.append((off, length, midi))
                    continue
                key = key_at.get(int(off), tune.home_key) if off.denominator == 1 else tune.home_key
                at = off
                for dur, m in play_sign(sign, key, midi, length):
                    expanded.append((at, dur, m))
                    at += dur
            notes = expanded
        played.append(notes)
    return played
