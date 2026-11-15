"""Extract Bach's own (S, A, T, B) chord at every beat of every four-part 4/4 chorale.

Training data for the Bach weights model, and the reference for the music metrics.
Writes .cache/bach_chords.json: {chorale id: {"key": [[tonic, mode] per beat], "chords": [[S, A, T, B] or null]}}.

    python scripts/extract_bach_chords.py
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chorale_engine import corpus  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / ".cache" / "bach_chords.json"


def part_notes(part):
    out = []
    for el in part.flatten().notesAndRests:
        length = Fraction(el.duration.quarterLength).limit_denominator(1 << 16)
        if length == 0 or not el.isNote:
            continue
        start = Fraction(el.offset).limit_denominator(1 << 16)
        out.append((start, start + length, int(el.pitch.midi)))
    return out


def sounding(notes, offset):
    for start, end, midi in notes:
        if start <= offset < end:
            return midi
    return None


def main() -> int:
    # Every chorale the beat extraction can handle, including the two deferred groups.
    corpus.REJECT_FINAL_OUTSIDE_TONIC = False
    corpus.REJECT_KEY_MAP_WITHOUT_HOME = False
    data = {}
    for cid in corpus.chorale_ids():
        tune, reason = corpus._build(corpus.load_raw(cid))
        if tune is None:
            continue
        parts = [part_notes(p) for p in corpus.parse_score(cid).parts]
        chords = []
        for beat in tune.beats:
            chord = [sounding(notes, beat.offset) for notes in parts]
            ok = all(v is not None for v in chord) and chord[0] == beat.soprano_midi
            chords.append(chord if ok else None)
        data[cid] = {
            "key": [[b.key.tonic, b.key.mode] for b in tune.beats],
            "cadence": [b.is_cadence for b in tune.beats],
            "chords": chords,
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, sort_keys=True))
    beats = sum(len(d["chords"]) for d in data.values())
    usable = sum(1 for d in data.values() for c in d["chords"] if c)
    print(f"{len(data)} chorales, {beats} beats, {usable} with a full four-part chord -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
