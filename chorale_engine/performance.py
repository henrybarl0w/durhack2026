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
from .ornaments import ornament_signs, ornaments, play_sign, suspensions

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
    parts = _segments(tune, chords, ornaments(chorale_id, chords) if ornamented else None,
                      suspensions(chorale_id, chords) if ornamented else None)
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


# --------------------------------------------------------------------------
# Timing: how a performer would pace the piece
# --------------------------------------------------------------------------

FERMATA_STRETCH = 1.6  # a phrase's last chord is held this much longer than written
FINAL_STRETCH = 2.2  # and the piece's last chord longer still
BREATH = 0.35  # quarter notes of silence after each phrase, before the next begins
RITARDANDO = (1.12, 1.28)  # the two beats before the final chord slow down by these factors


def timing(chorale_id: int):
    """A function from score position (quarter notes) to performance time (quarter notes at the base tempo).

    Fermata chords are held longer, each phrase ends with a breath, and the piece slows into its
    final chord. Multiply by 60 / bpm for seconds. It only stretches time: every note keeps its order.
    """
    tune = load_tune(chorale_id)
    last = len(tune.beats) - 1
    regions = []  # (start, end, stretch, pause after)
    for beat in tune.beats:
        start, end = Fraction(beat.offset), beat.offset + beat.length
        if beat.is_cadence:
            final = beat.index == last
            regions.append((start, end, FINAL_STRETCH if final else FERMATA_STRETCH, 0 if final else BREATH))
        elif beat.index in (last - 2, last - 1):
            regions.append((start, end, RITARDANDO[beat.index - (last - 2)], 0))
    regions.sort()

    def at(x, ending: bool = False) -> float:
        """ending=True for the end of a note: a note that ends where a phrase does stops before the breath."""
        x = Fraction(x)
        t = float(x)
        for start, end, stretch, pause in regions:
            if x <= start:
                break
            covered = min(x, end) - start
            t += float(covered) * (stretch - 1)
            if x > end or (x == end and not ending):
                t += pause
        return t

    return at
