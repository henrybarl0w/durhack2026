"""Recognise which chorale a score is from its soprano line, then decode it.

The decoder no longer needs the chorale ID. The soprano is Bach's own tune,
so it is compared, beat by beat, with every usable chorale. A chorale
matches when the top note on the upper staff equals its soprano at every one
of its beats, and its last note ends where the chorale's last soprano
note ends.

Some hymn tunes appear more than once in the corpus with the same soprano.
When several chorales match, each is tried as a decode. If every chorale
that decodes gives the same number, that number is the answer (reported with
the lowest chorale ID): some duplicates, such as 98 and 367, have identical
keys, openings and cadences, so they encode every number the same way.
Different numbers from different chorales raise DecodeError.
"""

from __future__ import annotations

from functools import lru_cache

from .codec import decode
from .corpus import load_tune
from .errors import DecodeError
from .musicxml_io import _staves, chord_from_staves, read_notes, staff_pitches_at
from .options import usable_chorales
from .weights import WeightsFn, default_weights


@lru_cache(maxsize=1)
def _signatures():
    """(chorale ID, end offset, ((offset, soprano MIDI), ...)) per usable chorale."""
    out = []
    for cid in usable_chorales():
        tune = load_tune(cid)
        end = max(e.offset + e.length for e in tune.soprano if e.midi is not None)
        out.append((cid, end, tuple((b.offset, b.soprano_midi) for b in tune.beats)))
    return tuple(out)


def _read(path):
    notes = read_notes(path)
    upper, lower = _staves(notes)
    end = max((n[3] + n[4] for n in notes), default=0)
    return upper, lower, end


def matching_chorales(path) -> list[int]:
    """Usable chorales whose soprano matches the score at every beat."""
    upper, _lower, end = _read(path)
    out = []
    for cid, cend, beats in _signatures():
        if cend != end:
            continue
        if all(
            (top := staff_pitches_at(upper, off)) and top[0] == midi
            for off, midi in beats
        ):
            out.append(cid)
    return out


def identify_chorale(path) -> int:
    """The chorale a score is written on. Raises DecodeError if not exactly one."""
    return decode_musicxml(path)[0]


def decode_musicxml(path, weights: WeightsFn = default_weights) -> tuple[int, int]:
    """(chorale ID, n) from a MusicXML score, with no chorale ID given."""
    candidates = matching_chorales(path)
    if not candidates:
        raise DecodeError("the soprano line matches no usable chorale")
    upper, lower, _end = _read(path)
    results = []
    first_error = None
    for cid in candidates:
        try:
            chords = [
                chord_from_staves(
                    staff_pitches_at(upper, b.offset),
                    staff_pitches_at(lower, b.offset),
                    f" at phrase {b.phrase_index}, beat {b.beat_index}",
                )
                for b in load_tune(cid).beats
            ]
            results.append((cid, decode(cid, chords, weights)))
        except DecodeError as exc:
            if first_error is None:
                first_error = exc
    if not results:
        raise first_error
    if len({n for _cid, n in results}) == 1:
        return results[0]
    raise DecodeError(
        "score decodes to different numbers on chorales with the same tune: "
        + ", ".join(f"{cid} -> {n}" for cid, n in results)
    )
