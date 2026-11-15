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

from dataclasses import dataclass, field
from functools import lru_cache

from .codec import decode
from .corpus import load_tune
from .errors import ChoraleRejected, DecodeError
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


def matching_chorales(path, tolerance: int = 0) -> list[int]:
    """Usable chorales whose soprano matches the score at every beat, or at all but `tolerance` beats."""
    upper, _lower, end = _read(path)
    out = []
    for cid, cend, beats in _signatures():
        if cend != end:
            continue
        misses = 0
        for off, midi in beats:
            top = staff_pitches_at(upper, off)
            if not top or top[0] != midi:
                misses += 1
                if misses > tolerance:
                    break
        if misses <= tolerance:
            out.append(cid)
    return out


def identify_chorale(path) -> int:
    """The chorale a score is written on. Raises DecodeError if not exactly one."""
    return decode_musicxml(path)[0]


def _chords_at_beats(upper, lower, cid, tolerant):
    """The chord at every beat of the chorale. tolerant: a beat that cannot be read is None."""
    chords = []
    for b in load_tune(cid).beats:
        try:
            chords.append(chord_from_staves(
                staff_pitches_at(upper, b.offset),
                staff_pitches_at(lower, b.offset),
                f" at phrase {b.phrase_index}, beat {b.beat_index}",
            ))
        except DecodeError:
            if not tolerant:
                raise
            chords.append(None)
    return chords


def _decode_with(path, weights, checksum: bool = False, max_errors: int | None = None):
    """(chorale ID, n, repaired (phrase, beat) list) from a MusicXML score with one weights function."""
    from .checksum import MAX_ERRORS, decode_checked

    if max_errors is None:
        max_errors = MAX_ERRORS if checksum else 0
    candidates = matching_chorales(path, tolerance=max_errors if checksum else 0)
    if not candidates:
        raise DecodeError("the soprano line matches no usable chorale")
    upper, lower, _end = _read(path)
    results = []
    first_error = None
    for cid in candidates:
        try:
            chords = _chords_at_beats(upper, lower, cid, tolerant=checksum)
            if checksum:
                repair = decode_checked(cid, chords, weights, max_errors)
                results.append((cid, repair.n, repair.fixed))
            else:
                results.append((cid, decode(cid, chords, weights), []))
        except (DecodeError, ChoraleRejected) as exc:
            if first_error is None:
                first_error = exc if isinstance(exc, DecodeError) else DecodeError(str(exc))
    if not results:
        raise first_error
    if len({n for _cid, n, _fixed in results}) == 1:
        return min(results, key=lambda r: len(r[2]))
    raise DecodeError(
        "score decodes to different numbers on chorales with the same tune: "
        + ", ".join(f"{cid} -> {n}" for cid, n, _fixed in results)
    )


@dataclass
class ScoreReading:
    chorale_id: int
    n: int
    style: str
    repaired: list = field(default_factory=list)  # (phrase, beat) of each misread chord that was fixed


def decode_score(path, checksum: bool = False, max_errors: int | None = None) -> ScoreReading:
    """Everything the decoder learns from a score: chorale, n, style, and any chords it repaired.

    checksum=True: the score was written with a checksum (encode_style(..., checksum=True));
    up to max_errors misread or unreadable chords are repaired (2 by default).
    """
    from .styles import STYLES, style_weights

    results, errors = [], []
    for style in STYLES:
        try:
            cid, n, fixed = _decode_with(path, style_weights(style), checksum, max_errors)
            results.append(ScoreReading(cid, n, style, fixed))
        except DecodeError as exc:
            errors.append(f"{style}: {exc}")
    if not results:
        raise DecodeError("the score decodes in no style (" + "; ".join(errors) + ")")
    if len({(r.chorale_id, r.n) for r in results}) > 1:
        raise DecodeError("the score decodes in more than one style: "
                          + ", ".join(f"{r.style} -> {r.n}" for r in results))
    return min(results, key=lambda r: len(r.repaired))


def decode_musicxml_with_style(path, checksum: bool = False) -> tuple[int, int, str]:
    """(chorale ID, n, style) from a MusicXML score: the style is worked out too.

    Every style is tried. A score decodes in the style it was written in; the chance that it
    also decodes in another is tiny, and if it ever did with a different number this raises
    DecodeError instead of guessing.
    """
    reading = decode_score(path, checksum)
    return reading.chorale_id, reading.n, reading.style


def decode_musicxml(path, weights: WeightsFn | None = None, checksum: bool = False) -> tuple[int, int]:
    """(chorale ID, n) from a MusicXML score, with no chorale ID given.

    With no weights the style (original or fancy) is worked out as well; pass weights to
    decode with exactly those. checksum=True for scores written with a checksum.
    """
    if weights is None:
        cid, n, _style = decode_musicxml_with_style(path, checksum)
        return cid, n
    cid, n, _fixed = _decode_with(path, weights, checksum)
    return cid, n
