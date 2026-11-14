"""Error correction on top of the mixed-radix phrase codec.

Why phrase-level?  A phrase is the natural unit of damage: each phrase starts
from a fixed start_state, so a wrong chord can never leak into the next phrase,
but inside a phrase it garbles the option lists of the following beats, so the
phrase's integer comes out as junk.  So we treat every phrase as one
"symbol column" that is either intact or lost, and use a Reed-Solomon style
erasure code across phrases to rebuild the lost ones.

Layout (r = parity_phrases):
  * Phrase p holds B_p = (capacity bits)//8 whole bytes: S_p = B_p-1 payload
    bytes plus 1 CRC-8 byte (over the phrase index and payload). The CRC (and
    the decoder noticing an impossible chord) is how a damaged phrase is spotted.
  * "Lane" j = the j-th payload byte of every phrase that has one. Each lane is
    a Reed-Solomon codeword over GF(256): k data bytes + r parity bytes, so any
    r damaged phrases (anywhere in the piece) can be rebuilt. This is MDS, i.e.
    optimal: r phrases of redundancy for r phrases of repair.
  * The user's integer n is split into the lanes' data bytes (little-endian).
"""
from __future__ import annotations

from .codec import _check_int, _decode_phrase, _encode_phrase, _phrase_tables
from .errors import CapacityError, ChordNotInOptions, DecodeError
from .options import get_plan
from .weights import WeightsFn, default_weights

# ---------------------------------------------------------------- GF(256)
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def _mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def _div(a, b):
    return 0 if a == 0 else _EXP[_LOG[a] - _LOG[b] + 255]


def _interp(points, x0):
    """Value at x0 of the unique polynomial through points [(x, y), ...]."""
    total = 0
    for i, (xi, yi) in enumerate(points):
        if yi == 0:
            continue
        num = den = 1
        for j, (xj, _) in enumerate(points):
            if j != i:
                num = _mul(num, x0 ^ xj)
                den = _mul(den, xi ^ xj)
        total ^= _mul(yi, _div(num, den))
    return total


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


# ----------------------------------------------------------------- layout
def _layout(plan, r):
    if type(r) is not int or r < 0:
        raise ValueError("parity_phrases must be a non-negative int")
    P = plan.tune.n_phrases
    caps = [_phrase_tables(plan, p)[0] for p in range(P)]
    S = [max((c.bit_length() - 1) // 8 - 1, 0) for c in caps]  # payload bytes per phrase
    lanes = [[p for p in range(P) if S[p] > j] for j in range(max(S, default=0))]
    if any(len(l) > 255 for l in lanes):
        raise ValueError("more than 255 phrases in a lane is not supported")
    ks = [max(len(l) - r, 0) for l in lanes]
    return S, lanes, ks


def ecc_capacity(chorale_id: int, parity_phrases: int = 2, weights: WeightsFn = default_weights) -> dict:
    """How much fits when `parity_phrases` damaged phrases must be survivable."""
    plan = get_plan(chorale_id, weights)
    _, _, ks = _layout(plan, parity_phrases)
    k = sum(ks)
    return {"bytes": k, "bits": 8 * k, "max_n": 256 ** k - 1}


# ----------------------------------------------------------------- encode
def _encode_plan(plan, n, r):
    S, lanes, ks = _layout(plan, r)
    K = sum(ks)
    if n < 0 or n >= 256 ** K:
        raise CapacityError(f"n={n} does not fit chorale {plan.chorale_id} with {r} parity phrases: "
                            f"capacity is {256 ** K} (0 <= n < 256**{K})")
    data = list(n.to_bytes(K, "little"))
    lane_vals, at = [], 0
    for lane, k in zip(lanes, ks):
        if k == 0:
            lane_vals.append([0] * len(lane))
            continue
        d = data[at:at + k]
        at += k
        pts = list(enumerate(d))
        lane_vals.append(d + [_interp(pts, x) for x in range(k, len(lane))])
    chords = []
    for p in range(plan.tune.n_phrases):
        if S[p] == 0:
            chords.extend(_encode_phrase(plan, p, 0))
            continue
        payload = bytes(lane_vals[j][lanes[j].index(p)] for j in range(S[p]))
        block = payload + bytes([_crc8(bytes([p & 255]) + payload)])
        chords.extend(_encode_phrase(plan, p, int.from_bytes(block, "little")))
    return chords


def encode_ecc(chorale_id: int, n: int, parity_phrases: int = 2,
               weights: WeightsFn = default_weights) -> list:
    """Chords encoding n, surviving up to `parity_phrases` damaged phrases."""
    plan = get_plan(chorale_id, weights)
    return _encode_plan(plan, _check_int(n), parity_phrases)


# ----------------------------------------------------------------- decode
def decode_ecc_report(chorale_id: int, chords, parity_phrases: int = 2,
                      weights: WeightsFn = default_weights):
    """Returns (n, repaired_chords, damaged_phrase_indices)."""
    plan = get_plan(chorale_id, weights)
    r = parity_phrases
    chords = list(chords)
    if len(chords) != len(plan.tune.beats):
        raise DecodeError(f"chorale {chorale_id} has {len(plan.tune.beats)} beats, got {len(chords)} chords")
    S, lanes, ks = _layout(plan, r)
    where = [{p: i for i, p in enumerate(l)} for l in lanes]
    vals = [[None] * len(l) for l in lanes]  # None = erased
    bad = []
    for p, (start, end) in enumerate(plan.tune.phrases):
        if S[p] == 0:
            continue
        try:
            n_p = _decode_phrase(plan, p, chords[start:end])
        except (DecodeError, ChordNotInOptions, ValueError, TypeError):
            n_p = None
        block = None
        if n_p is not None and n_p < 256 ** (S[p] + 1):
            block = n_p.to_bytes(S[p] + 1, "little")
            if _crc8(bytes([p & 255]) + block[:-1]) != block[-1]:
                block = None
        if block is None:
            bad.append(p)
            continue
        for j in range(S[p]):
            vals[j][where[j][p]] = block[j]

    data = []
    for j, (lane, k) in enumerate(zip(lanes, ks)):
        if k == 0:
            continue
        good = [(x, v) for x, v in enumerate(vals[j]) if v is not None]
        if len(good) < k:
            raise DecodeError(f"too many damaged phrases {bad}: only {r} can be repaired")
        basis = good[:k]
        for x, v in good[k:]:  # spare redundancy: use it to catch undetected damage
            if _interp(basis, x) != v:
                raise DecodeError(f"inconsistent data in lane {j}: damage the CRC did not catch")
        data.extend(vals[j][x] if vals[j][x] is not None else _interp(basis, x) for x in range(k))
    n = int.from_bytes(bytes(data), "little")
    return n, _encode_plan(plan, n, r), bad


def decode_ecc(chorale_id: int, chords, parity_phrases: int = 2,
               weights: WeightsFn = default_weights) -> int:
    return decode_ecc_report(chorale_id, chords, parity_phrases, weights)[0]


# ------------------------------------------------------- word-list messages
def best_parity(chorale_id: int, needed_bytes: int, max_parity: int = 2, min_parity: int = 1,
                weights: WeightsFn = default_weights):
    """Most repairable phrases (max_parity..min_parity) that still leave room for needed_bytes, else None."""
    for r in range(max_parity, min_parity - 1, -1):
        if ecc_capacity(chorale_id, r, weights)["bytes"] >= needed_bytes:
            return r
    return None


def pack_indices(indices, vocab_size: int) -> int:
    """Word-list indices -> one integer. Each index becomes a digit 1..vocab_size in base vocab_size+1
    (digit 0 never appears, so the number of words is recoverable)."""
    base = vocab_size + 1
    n = 0
    for i, idx in enumerate(indices):
        if not 0 <= idx < vocab_size:
            raise ValueError(f"word index {idx} is outside the word list")
        n += (idx + 1) * base ** i
    return n


def unpack_indices(n: int, vocab_size: int) -> list:
    base = vocab_size + 1
    out = []
    while n:
        n, digit = divmod(n, base)
        if digit == 0:
            raise DecodeError("the decoded number is not a valid word sequence")
        out.append(digit - 1)
    return out


def encode_words(chorale_id: int, indices, vocab_size: int, parity_phrases: int = 2,
                 weights: WeightsFn = default_weights) -> list:
    """Chords carrying a sequence of word-list indices, with error correction."""
    return encode_ecc(chorale_id, pack_indices(indices, vocab_size), parity_phrases, weights)


def decode_words(chorale_id: int, chords, vocab_size: int, parity_phrases: int = 2,
                 weights: WeightsFn = default_weights):
    """Returns (word indices, repaired chords, damaged phrase indices)."""
    n, repaired, bad = decode_ecc_report(chorale_id, chords, parity_phrases, weights)
    return unpack_indices(n, vocab_size), repaired, bad