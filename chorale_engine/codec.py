"""Mixed-radix encode and decode, per phrase and whole piece, and capacity.

Everything goes through Plan.options(), the same function the public
options() uses, so the encoder, the decoder and the capacity agree.
All of n, digits, counts and capacities are Python ints.
"""

from __future__ import annotations

from .errors import CapacityError, ChordNotInOptions, DecodeError
from .options import _arrival, get_plan
from .weights import WeightsFn, default_weights


def _check_int(n, what="n") -> int:
    if type(n) is not int:
        raise TypeError(f"{what} must be a Python int, got {type(n).__name__}")
    return n


def _state_key(plan, prev, ppb):
    """States that are guaranteed to share option lists from here on."""
    if prev is None:
        return (None, None)
    if plan.weights is default_weights:
        return (prev, _arrival(ppb, prev[3]))
    return (prev, ppb)


def _phrase_tables(plan, p: int):
    """(capacity, per-beat minimum counts) of phrase p, cached on the plan."""
    cache = plan.__dict__.setdefault("_capacity_cache", {})
    if p in cache:
        return cache[p]
    start, end = plan.tune.phrases[p]
    length = end - start

    # Forward: every state the encoder can be in at each beat.
    layers = []
    states = {_state_key(plan, *plan.start_state(p)): plan.start_state(p)}
    for b in range(length):
        layer = {}
        nxt = {}
        for key in sorted(states, key=repr):
            prev, ppb = states[key]
            opts = plan.options(p, b, prev, ppb)
            succ = []
            for c in opts:
                state = (c, prev[3] if prev is not None else None)
                k = _state_key(plan, *state)
                nxt.setdefault(k, state)
                succ.append(k)
            layer[key] = (len(opts), succ)
        layers.append(layer)
        states = nxt

    # Backward: the minimum product of counts from each state to the end.
    best = {k: 1 for k in states}
    for b in range(length - 1, -1, -1):
        best = {
            key: count * min(best[k] for k in succ)
            for key, (count, succ) in layers[b].items()
        }
    (capacity,) = best.values()
    min_counts = [min(count for count, _succ in layer.values()) for layer in layers]
    cache[p] = (capacity, min_counts)
    return cache[p]


def phrase_capacity(chorale_id: int, phrase_index: int, weights: WeightsFn = default_weights) -> int:
    """Cap_p: n is encodable in the phrase exactly when 0 <= n < Cap_p."""
    plan = get_plan(chorale_id, weights)
    plan.start_state(phrase_index)  # range check
    return _phrase_tables(plan, phrase_index)[0]


def capacity_report(chorale_id: int, weights: WeightsFn = default_weights) -> dict:
    """Per-phrase capacity, bits and per-beat minimum counts, and the total."""
    plan = get_plan(chorale_id, weights)
    phrases = []
    total = 1
    for p in range(plan.tune.n_phrases):
        cap, min_counts = _phrase_tables(plan, p)
        total *= cap
        phrases.append({
            "phrase_index": p,
            "capacity": cap,
            "bits": cap.bit_length() - 1,
            "min_counts": list(min_counts),
        })
    return {
        "chorale_id": chorale_id,
        "phrases": phrases,
        "capacity": total,
        "bits": total.bit_length() - 1,
    }


def _encode_phrase(plan, p: int, n: int) -> list:
    cap = _phrase_tables(plan, p)[0]
    if n < 0 or n >= cap:
        raise CapacityError(
            f"n={n} does not fit phrase {p} of chorale {plan.chorale_id}: "
            f"capacity is {cap} (0 <= n < {cap})"
        )
    start, end = plan.tune.phrases[p]
    prev, ppb = plan.start_state(p)
    chords = []
    for b in range(end - start):
        opts = plan.options(p, b, prev, ppb)
        digit = n % len(opts)
        n //= len(opts)
        chord = opts[digit]
        chords.append(chord)
        prev, ppb = chord, (prev[3] if prev is not None else None)
    if n != 0:  # cannot happen when n < Cap_p
        raise CapacityError(f"n does not fit phrase {p}: capacity is {cap}")
    return chords


def _decode_phrase(plan, p: int, chords) -> int:
    start, end = plan.tune.phrases[p]
    chords = [tuple(int(m) for m in c) for c in chords]
    if len(chords) != end - start:
        raise DecodeError(
            f"phrase {p} of chorale {plan.chorale_id} has {end - start} beats, got {len(chords)} chords"
        )
    prev, ppb = plan.start_state(p)
    digits = []
    for b, chord in enumerate(chords):
        opts = plan.options(p, b, prev, ppb)
        if chord not in opts:
            raise ChordNotInOptions(p, b, chord, list(opts))
        digits.append((opts.index(chord), len(opts)))
        prev, ppb = chord, (prev[3] if prev is not None else None)
    n = 0
    for digit, count in reversed(digits):
        n = n * count + digit
    cap = _phrase_tables(plan, p)[0]
    if n >= cap:
        raise DecodeError(f"phrase {p} decodes to {n}, which is not below its capacity {cap}")
    return n


def encode_phrase(chorale_id: int, phrase_index: int, n: int,
                  weights: WeightsFn = default_weights) -> list:
    """Chords of one phrase that encode n (0 <= n < phrase_capacity)."""
    plan = get_plan(chorale_id, weights)
    plan.start_state(phrase_index)
    return _encode_phrase(plan, phrase_index, _check_int(n))


def decode_phrase(chorale_id: int, phrase_index: int, chords,
                  weights: WeightsFn = default_weights) -> int:
    """The integer written in one phrase's chords."""
    plan = get_plan(chorale_id, weights)
    plan.start_state(phrase_index)
    return _decode_phrase(plan, phrase_index, chords)


def encode(chorale_id: int, n: int, weights: WeightsFn = default_weights) -> list:
    """Chords of the whole piece that encode n. Convenience and tests only."""
    plan = get_plan(chorale_id, weights)
    n = _check_int(n)
    caps = [_phrase_tables(plan, p)[0] for p in range(plan.tune.n_phrases)]
    total = 1
    for cap in caps:
        total *= cap
    if n < 0 or n >= total:
        raise CapacityError(
            f"n={n} does not fit chorale {chorale_id}: capacity is {total} (0 <= n < {total})"
        )
    chords = []
    for p, cap in enumerate(caps):
        chords.extend(_encode_phrase(plan, p, n % cap))
        n //= cap
    return chords


def decode(chorale_id: int, chords, weights: WeightsFn = default_weights) -> int:
    """The integer written in the whole piece's chords."""
    plan = get_plan(chorale_id, weights)
    chords = list(chords)
    if len(chords) != len(plan.tune.beats):
        raise DecodeError(
            f"chorale {chorale_id} has {len(plan.tune.beats)} beats, got {len(chords)} chords"
        )
    n = 0
    scale = 1
    for p, (start, end) in enumerate(plan.tune.phrases):
        n += _decode_phrase(plan, p, chords[start:end]) * scale
        scale *= _phrase_tables(plan, p)[0]
    return n
