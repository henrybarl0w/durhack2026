"""Checksum and repair: a 16-bit CRC on the message, and a decoder that fixes misread chords.

The phrase-level error correction in ecc.py needs whole bytes per phrase, and phrases hold
only 5-12 bits, so it never fits. This uses the music itself to find errors instead:

  * Encoding stores n * 2**16 + crc16(n): 16 bits of capacity buy a fingerprint of n.
  * Decoding checks the fingerprint. If a chord was misread, it is usually not one of that
    beat's (at most 4) options, which points at the beat; a misread that happens to be a
    legal option makes a later beat of the same phrase fail instead, which bounds where it
    can be. The decoder tries the other options at those beats (and, if nothing points
    anywhere, at every beat), and keeps the result whose fingerprint matches.
  * Phrases cannot pass an error on: each starts from its own fixed state.

Up to max_errors (default 2) chords are repaired. A beat that could not be read at all can
be passed as None and is treated the same way. Beats with only one option (cadences, a fixed
opening) are corrected for free. With 16 bits, a wrong repair passing by chance is about
1 in 2,700 for one error and 1 in 110 for two, and a misreading that decodes cleanly but
wrongly is caught instead of being shown as the message.

    chords = encode_checked(cid, n)
    result = decode_checked(cid, chords)      # Repair(n, chords, fixed=[(phrase, beat), ...])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .codec import _check_int, _phrase_tables, encode
from .errors import CapacityError, DecodeError
from .options import get_plan
from .weights import WeightsFn, default_weights

CHECK_BITS = 16
MAX_ERRORS = 2


def crc16(n: int) -> int:
    """CRC-16/CCITT-FALSE of n's big-endian bytes (at least one byte)."""
    data = n.to_bytes(max(1, (n.bit_length() + 7) // 8), "big")
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def with_check(n: int) -> int:
    """n with its checksum attached: the number that goes into the music."""
    n = _check_int(n)
    if n < 0:
        raise CapacityError(f"n={n} is negative")
    return (n << CHECK_BITS) | crc16(n)


def split_check(value: int) -> int | None:
    """n if value carries a correct checksum, else None."""
    n, check = value >> CHECK_BITS, value & ((1 << CHECK_BITS) - 1)
    return n if crc16(n) == check else None


def checked_capacity(chorale_id: int, weights: WeightsFn = default_weights) -> int:
    """n fits with a checksum exactly when 0 <= n < this."""
    from .codec import capacity_report

    return capacity_report(chorale_id, weights)["capacity"] >> CHECK_BITS


def encode_checked(chorale_id: int, n: int, weights: WeightsFn = default_weights) -> list:
    """Chords encoding n with its checksum."""
    cap = checked_capacity(chorale_id, weights)
    if _check_int(n) < 0 or n >= cap:
        raise CapacityError(f"n={n} does not fit chorale {chorale_id} with a checksum: capacity is {cap}")
    return encode(chorale_id, with_check(n), weights)


@dataclass
class Repair:
    n: int
    chords: list  # the chords as they must have been written
    fixed: list = field(default_factory=list)  # (phrase, beat) of every chord that was repaired


# --------------------------------------------------------------------------
# phrase-level decoding that tolerates single-option beats and reports where it stops
# --------------------------------------------------------------------------

def _walk(plan, p, chords):
    """Decode phrase p. Returns (value, chords as decoded, None) or (None, None, first bad beat)."""
    start, end = plan.tune.phrases[p]
    prev, ppb = plan.start_state(p)
    digits, out = [], []
    for b in range(end - start):
        opts = plan.options(p, b, prev, ppb)
        chord = chords[b]
        if chord is not None and tuple(chord) in opts:
            chord = tuple(chord)
        elif len(opts) == 1:
            chord = opts[0]  # a beat with one option carries no information: always right
        else:
            return None, None, b
        digits.append((opts.index(chord), len(opts)))
        out.append(chord)
        prev, ppb = chord, (prev[3] if prev is not None else None)
    value = 0
    for digit, count in reversed(digits):
        value = value * count + digit
    if value >= _phrase_tables(plan, p)[0]:
        return None, None, end - start - 1
    return value, out, None


def _alternatives(plan, p, chords, window, depth):
    """Phrase values reachable by replacing `depth` chords: {value: (chords, fixed beats)}.

    The first replaced beat lies in `window` (no later than the first beat that failed to
    decode); any further ones can be anywhere after it, since a second error in the same
    phrase may come after the point where the first one showed.
    """
    length = len(chords)
    found = {}
    for first in window:
        for rest in combinations(range(first + 1, length), depth - 1):
            _search(plan, p, list(chords), [first, *rest], [], found)
    return found


def _search(plan, p, chords, todo, fixed, found):
    """Replace the chord at each beat in todo (in order) by every other option there, then decode."""
    if not todo:
        value, out, _bad = _walk(plan, p, chords)
        if value is not None and value not in found:
            free = [b for b, (a, c) in enumerate(zip(chords, out)) if b not in fixed and (a is None or tuple(a) != c)]
            found[value] = (out, sorted(fixed + free))
        return
    b = todo[0]
    prev, ppb = plan.start_state(p)
    for i in range(b):  # the state at beat b, from the chords before it
        opts = plan.options(p, i, prev, ppb)
        chord = chords[i]
        if chord is None or tuple(chord) not in opts:
            if len(opts) == 1:
                chord = opts[0]
            else:
                return  # an earlier beat is wrong too: not this combination
        prev, ppb = tuple(chord), (prev[3] if prev is not None else None)
    for option in plan.options(p, b, prev, ppb):
        if chords[b] is not None and option == tuple(chords[b]):
            continue
        trial = list(chords)
        trial[b] = option
        _search(plan, p, trial, todo[1:], fixed + [b], found)


def decode_checked(chorale_id: int, chords, weights: WeightsFn = default_weights,
                   max_errors: int = MAX_ERRORS) -> Repair:
    """n from chords written with encode_checked, repairing up to max_errors misread chords.

    chords may contain None for beats that could not be read. Raises DecodeError when no
    repair within max_errors gives a correct checksum, or when two different repairs do.
    """
    plan = get_plan(chorale_id, weights)
    chords = list(chords)
    if len(chords) != len(plan.tune.beats):
        raise DecodeError(f"chorale {chorale_id} has {len(plan.tune.beats)} beats, got {len(chords)} chords")
    phrases = plan.tune.phrases
    caps = [_phrase_tables(plan, p)[0] for p in range(len(phrases))]
    pieces = [chords[s:e] for s, e in phrases]

    # Level 0: each phrase as read. A phrase that fails points at a window of beats.
    as_read, windows = [], []
    for p, piece in enumerate(pieces):
        value, out, bad = _walk(plan, p, piece)
        free = [] if out is None else [b for b, (a, c) in enumerate(zip(piece, out)) if a is None or tuple(a) != c]
        as_read.append(None if value is None else {value: (out, free)})
        windows.append(list(range(len(piece))) if bad is None else list(range(bad + 1)))
    suspects = [p for p, level in enumerate(as_read) if level is None]
    if len(suspects) > max_errors:
        raise DecodeError(f"too many phrases are damaged ({len(suspects)}): retake the photo")

    cache = {}

    def level(p, depth):
        if depth == 0:
            return as_read[p] or {}
        if (p, depth) not in cache:
            cache[(p, depth)] = _alternatives(plan, p, pieces[p], windows[p], depth)
        return cache[(p, depth)]

    def combine(choice):
        """Every whole-piece value from one value per phrase, at the depths in `choice`."""
        results = [(0, 1, [], [])]  # (value so far, scale, chords, fixed)
        for p, depth in enumerate(choice):
            options = level(p, depth)
            if not options:
                return []
            results = [
                (value + v * scale, scale * caps[p], out + c, fixed + [(p, b) for b in f])
                for value, scale, out, fixed in results
                for v, (c, f) in options.items()
            ]
        return results

    for total in range(max_errors + 1):
        passing = {}
        for choice in _depths(len(phrases), suspects, total):
            for value, _scale, out, fixed in combine(choice):
                n = split_check(value)
                if n is not None and n not in passing:
                    passing[n] = Repair(n, out, fixed)
        if len(passing) == 1:
            return next(iter(passing.values()))
        if len(passing) > 1:
            raise DecodeError("the score could be read more than one way: retake the photo")
    raise DecodeError(f"the message could not be repaired within {max_errors} chord(s): retake the photo")


def _depths(n_phrases, suspects, total):
    """Ways to spend exactly `total` repairs over the phrases; every suspect phrase gets at least one."""
    def place(p, left):
        if p == n_phrases:
            if left == 0:
                yield []
            return
        low = 1 if p in suspects else 0
        for d in range(low, left + 1):
            for rest in place(p + 1, left - d):
                yield [d] + rest
    yield from place(0, total)
