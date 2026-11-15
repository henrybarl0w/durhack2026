"""Write a MIDI file of an encoded chorale, playing the ornaments as a performer would.

    python scripts/listen.py --chorale 16 --n 12345 --out chorale16.mid --bach --ornaments --signs

--bach uses the Bach-trained weights, --strict adds the textbook style rules, --ornaments adds the off-beat quavers, --signs plays the
trills, turns and mordents (the score itself only shows their signs).
"""

from __future__ import annotations

import argparse
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import music21  # noqa: E402

import chorale_engine as ce  # noqa: E402
from chorale_engine.bach_weights import bach_weights  # noqa: E402
from chorale_engine.style import strict_weights  # noqa: E402
from chorale_engine.musicxml_io import _segments  # noqa: E402
from chorale_engine.ornaments import ornament_signs, ornaments, play_sign  # noqa: E402

_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def midi_of(pitch):
    step, alter, octave = pitch
    return 12 * (octave + 1) + _STEP_PC[step] + alter


def performance(chorale_id, chords, with_ornaments, with_signs):
    """Per voice: [(offset, duration, MIDI)] as played."""
    tune = ce.load_tune(chorale_id)
    parts = _segments(tune, chords, ornaments(chorale_id, chords) if with_ornaments else None)
    signs = {}
    if with_signs:
        signs = {tune.soprano[i].offset: name for i, name in ornament_signs(chorale_id).items()}
    key_at = {}
    for beat in tune.beats:
        key_at[beat.offset] = beat.key
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
        if v == 0 and signs:
            expanded = []
            for off, length, midi in notes:
                sign = signs.get(off)
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


def write_midi(played, path, tempo=72):
    score = music21.stream.Score()
    for v, notes in enumerate(played):
        part = music21.stream.Part()
        part.insert(0, music21.instrument.Organ())
        if v == 0:
            part.insert(0, music21.tempo.MetronomeMark(number=tempo))
        for off, dur, midi in notes:
            n = music21.note.Note(midi)
            n.quarterLength = float(dur)
            part.insert(float(off), n)
        score.insert(0, part)
    score.write("midi", str(path))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chorale", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bach", action="store_true")
    parser.add_argument("--strict", action="store_true", help="the strict style (implies the Bach model)")
    parser.add_argument("--ornaments", action="store_true")
    parser.add_argument("--signs", action="store_true")
    parser.add_argument("--tempo", type=int, default=72)
    args = parser.parse_args(argv)
    weights = strict_weights if args.strict else bach_weights if args.bach else None
    chords = ce.encode(args.chorale, args.n, weights) if weights else ce.encode(args.chorale, args.n)
    write_midi(performance(args.chorale, chords, args.ornaments, args.signs), args.out, args.tempo)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
