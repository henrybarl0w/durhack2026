"""Write a MIDI file of an encoded chorale, playing the ornaments as a performer would.

    python scripts/listen.py --chorale 16 --n 12345 --out chorale16.mid --bach --ornaments --signs

--bach uses the Bach-trained weights, --strict adds the textbook style rules, --ornaments adds the off-beat quavers, --signs plays the
trills, turns and mordents (the score itself only shows their signs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import music21  # noqa: E402

import chorale_engine as ce  # noqa: E402
from chorale_engine.bach_weights import bach_weights  # noqa: E402
from chorale_engine.style import strict_weights  # noqa: E402
from chorale_engine.performance import performance  # noqa: E402

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
    write_midi(performance(args.chorale, chords, ornamented=args.ornaments, signs=args.signs), args.out, args.tempo)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
