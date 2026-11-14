"""Milestone 4: mixed-radix encode/decode and capacity."""

import itertools
import random

import pytest

from chorale_engine import (
    CapacityError,
    ChordNotInOptions,
    DecodeError,
    capacity_report,
    decode,
    decode_phrase,
    encode,
    encode_phrase,
    load_tune,
    options,
    phrase_capacity,
    start_state,
)
from chorale_engine.weights import default_weights

SAMPLE = [6, 14, 16, 47, 58, 227]


def digits_of(cid, p, chords, weights=default_weights):
    """(digit, count) per beat, by replaying the public options()."""
    prev, ppb = start_state(cid, p, weights)
    out = []
    for b, chord in enumerate(chords):
        opts = options(cid, p, b, prev, ppb, weights)
        out.append((opts.index(chord), len(opts)))
        prev, ppb = chord, (prev[3] if prev is not None else None)
    return out


@pytest.mark.parametrize("cid", SAMPLE)
def test_phrase_round_trip_at_the_edges(cid):
    tune = load_tune(cid)
    rng = random.Random(cid)
    for p in range(tune.n_phrases):
        cap = phrase_capacity(cid, p)
        assert type(cap) is int and cap >= 1
        values = {0, cap - 1, cap // 2, rng.randrange(cap), rng.randrange(cap)}
        for n in sorted(values):
            chords = encode_phrase(cid, p, n)
            assert len(chords) == len(tune.phrase_beats(p))
            assert decode_phrase(cid, p, chords) == n
        for bad in (cap, cap + 1, -1, cap * 1000):
            with pytest.raises(CapacityError) as err:
                encode_phrase(cid, p, bad)
            assert str(cap) in str(err.value)


@pytest.mark.parametrize("cid", SAMPLE)
def test_whole_piece_round_trip_and_capacity_errors(cid):
    tune = load_tune(cid)
    report = capacity_report(cid)
    cap = report["capacity"]
    assert type(cap) is int
    rng = random.Random(cid + 1)
    for n in [0, 1, cap - 1] + [rng.randrange(cap) for _ in range(5)]:
        chords = encode(cid, n)
        assert len(chords) == len(tune.beats)
        assert decode(cid, chords) == n
    for bad in (cap, -1, cap + 12345):
        with pytest.raises(CapacityError) as err:
            encode(cid, bad)
        assert str(cap) in str(err.value)
    with pytest.raises(TypeError):
        encode(cid, 1.0)


def test_capacity_report_shape():
    cid = 6
    tune = load_tune(cid)
    report = capacity_report(cid)
    total = 1
    for p, row in enumerate(report["phrases"]):
        assert row["phrase_index"] == p
        assert row["capacity"] == phrase_capacity(cid, p)
        assert row["bits"] == row["capacity"].bit_length() - 1
        assert 2 ** row["bits"] <= row["capacity"] < 2 ** (row["bits"] + 1)
        assert len(row["min_counts"]) == len(tune.phrase_beats(p))
        assert all(1 <= c <= 4 for c in row["min_counts"])
        assert row["min_counts"][-1] == 1  # the cadence beat
        total *= row["capacity"]
    assert report["phrases"][0]["min_counts"][0] == 1  # fixed opening
    assert report["capacity"] == total
    assert report["bits"] == total.bit_length() - 1


def test_capacities_can_exceed_64_bits():
    report = capacity_report(215)
    assert report["capacity"] > 2 ** 64
    n = report["capacity"] - 1
    assert decode(215, encode(215, n)) == n


def _all_paths(cid, p):
    """Every path the encoder can take through a phrase: (chords, product of counts)."""
    tune = load_tune(cid)
    length = len(tune.phrase_beats(p))
    out = []

    def go(b, prev, ppb, chords, product):
        if b == length:
            out.append((tuple(chords), product))
            return
        opts = options(cid, p, b, prev, ppb)
        for c in opts:
            go(b + 1, c, prev[3] if prev is not None else None, chords + [c], product * len(opts))

    go(0, *start_state(cid, p), [], 1)
    return out


@pytest.mark.parametrize("cid,p", [(14, 2), (19, 3), (86, 3), (6, 1)])
def test_capacity_is_the_minimum_product_over_all_paths(cid, p):
    paths = _all_paths(cid, p)
    cap = phrase_capacity(cid, p)
    assert cap == min(product for _chords, product in paths)
    # Exactly the values below Cap_p encode, and each to a different path.
    seen = set()
    for n in range(cap):
        chords = tuple(encode_phrase(cid, p, n))
        assert chords in {c for c, _ in paths}
        assert chords not in seen
        seen.add(chords)
        assert decode_phrase(cid, p, chords) == n
    # Per-beat minimum counts agree with the enumeration.
    report = capacity_report(cid)["phrases"][p]
    by_beat = [set() for _ in report["min_counts"]]
    for chords, _product in paths:
        for b, (_digit, count) in enumerate(digits_of(cid, p, chords)):
            by_beat[b].add(count)
    assert report["min_counts"] == [min(s) for s in by_beat]


@pytest.mark.parametrize("cid", SAMPLE)
def test_every_encoded_chord_is_the_option_at_its_digit(cid):
    tune = load_tune(cid)
    rng = random.Random(cid + 2)
    for p in range(tune.n_phrases):
        cap = phrase_capacity(cid, p)
        for n in {0, cap - 1, rng.randrange(cap)}:
            chords = encode_phrase(cid, p, n)
            rest = n
            prev, ppb = start_state(cid, p)
            for b, chord in enumerate(chords):
                opts = options(cid, p, b, prev, ppb)
                digit = rest % len(opts)
                rest //= len(opts)
                assert chord in opts and opts.index(chord) == digit, (cid, p, b)
                prev, ppb = chord, (prev[3] if prev is not None else None)
            assert rest == 0
    # n = 0 picks option 0 everywhere.
    assert all(d == 0 for d, _c in digits_of(cid, 0, encode_phrase(cid, 0, 0)))


@pytest.mark.parametrize("cid", [6, 16, 58])
def test_phrases_are_independent(cid):
    tune = load_tune(cid)
    caps = [phrase_capacity(cid, p) for p in range(tune.n_phrases)]
    rng = random.Random(cid + 3)
    chunks = [rng.randrange(c) for c in caps]

    def whole(values):
        n = 0
        for value, cap in zip(reversed(values), reversed(caps)):
            n = n * cap + value
        return n

    base = encode(cid, whole(chunks))
    for p, (start, end) in enumerate(tune.phrases):
        assert base[start:end] == encode_phrase(cid, p, chunks[p])
    for p in range(tune.n_phrases):
        if caps[p] == 1:
            continue
        changed = list(chunks)
        changed[p] = (chunks[p] + 1) % caps[p]
        other = encode(cid, whole(changed))
        for q, (start, end) in enumerate(tune.phrases):
            if q == p:
                assert other[start:end] != base[start:end]
            else:
                assert other[start:end] == base[start:end], (p, q)
        assert decode(cid, other) == whole(changed)


def _alter(cid, p, b, chord):
    """A different chord with the same soprano that is not an option at that beat."""
    prev_state = None
    for delta in (1, -1, 2, -2, 3, -3):
        cand = (chord[0], chord[1], chord[2], chord[3] + delta)
        if 40 <= cand[3] <= 60:
            return cand
    raise AssertionError("no altered chord found")


@pytest.mark.parametrize("cid", [6, 16, 58])
def test_altered_chord_raises_chord_not_in_options(cid):
    tune = load_tune(cid)
    rng = random.Random(cid + 4)
    caps = [phrase_capacity(cid, p) for p in range(tune.n_phrases)]
    for _ in range(6):
        p = rng.randrange(tune.n_phrases)
        chords = encode_phrase(cid, p, rng.randrange(caps[p]))
        b = rng.randrange(len(chords))
        prev, ppb = start_state(cid, p)
        for i in range(b):
            prev, ppb = chords[i], (prev[3] if prev is not None else None)
        opts = options(cid, p, b, prev, ppb)
        wrong = _alter(cid, p, b, chords[b])
        while wrong in opts:
            wrong = (wrong[0], wrong[1], wrong[2] - 1, wrong[3])
        broken = list(chords)
        broken[b] = wrong
        with pytest.raises(ChordNotInOptions) as err:
            decode_phrase(cid, p, broken)
        exc = err.value
        assert isinstance(exc, DecodeError)
        assert (exc.phrase_index, exc.beat_index) == (p, b)
        assert exc.chord == wrong and exc.options == opts
        assert f"phrase {p}" in str(exc) and f"beat {b}" in str(exc)
        # The same through the whole-piece decoder.
        piece = encode(cid, 0)
        start = tune.phrases[p][0]
        piece[start:start + len(chords)] = broken
        with pytest.raises(ChordNotInOptions) as err:
            decode(cid, piece)
        assert (err.value.phrase_index, err.value.beat_index) == (p, b)


def test_cadence_and_opening_beats_are_checked_too():
    cid = 6
    chords = encode_phrase(cid, 0, 0)
    last = len(chords) - 1
    broken = list(chords)
    broken[last] = (chords[last][0], chords[last][1], chords[last][2], chords[last][3] + 12)
    with pytest.raises(ChordNotInOptions) as err:
        decode_phrase(cid, 0, broken)
    assert (err.value.phrase_index, err.value.beat_index) == (0, last)
    assert err.value.options == [chords[last]]
    broken = list(chords)
    broken[0] = (chords[0][0], chords[0][1], chords[0][2], chords[0][3] + 12)
    with pytest.raises(ChordNotInOptions) as err:
        decode_phrase(cid, 0, broken)
    assert (err.value.phrase_index, err.value.beat_index) == (0, 0)
    assert err.value.options == [chords[0]]


def test_wrong_number_of_chords_raises_decode_error():
    cid = 6
    chords = encode_phrase(cid, 1, 0)
    for bad in (chords[:-1], chords + [chords[-1]], []):
        with pytest.raises(DecodeError) as err:
            decode_phrase(cid, 1, bad)
        assert not isinstance(err.value, ChordNotInOptions)
    piece = encode(cid, 0)
    for bad in (piece[:-1], piece + [piece[-1]]):
        with pytest.raises(DecodeError) as err:
            decode(cid, bad)
        assert not isinstance(err.value, ChordNotInOptions)


def test_value_at_or_above_capacity_is_a_decode_error():
    # Find a legal path whose product of counts exceeds Cap_p, and take its top digits.
    found = False
    for cid, p in [(14, 2), (19, 3), (86, 3), (6, 1)]:
        cap = phrase_capacity(cid, p)
        for chords, product in _all_paths(cid, p):
            if product <= cap:
                continue
            value = 0
            for digit, count in reversed(digits_of(cid, p, chords)):
                value = value * count + digit
            if value >= cap:
                with pytest.raises(DecodeError) as err:
                    decode_phrase(cid, p, list(chords))
                assert not isinstance(err.value, ChordNotInOptions)
                assert f"phrase {p}" in str(err.value)
                found = True
                break
        if found:
            break
    assert found


def test_lists_and_tuples_decode_alike():
    cid = 6
    chords = encode(cid, 12345)
    assert decode(cid, [list(c) for c in chords]) == 12345


def test_custom_weights_round_trip():
    def reversed_weights(state, opts):
        return [-w for w in default_weights(state, opts)]

    cid = 6
    cap = capacity_report(cid, reversed_weights)["capacity"]
    n = cap - 1
    chords = encode(cid, n, reversed_weights)
    assert decode(cid, chords, reversed_weights) == n
    assert chords != encode(cid, min(n, capacity_report(cid)["capacity"] - 1))
