"""Milestone 3: weights."""

import copy

import pytest

from chorale_engine.keys import Key
from chorale_engine.weights import (
    WeightState,
    checked_weights,
    chord_root_pc,
    default_weights,
    pair_weight,
)

C = Key("C", "major")


def state(prev, ppb=None, **kw):
    base = dict(chorale_id=6, phrase_index=0, beat_index=1, key=C, soprano_midi=67,
                prev_chord=prev, prev_prev_bass=ppb)
    base.update(kw)
    return WeightState(**base)


I_ROOT = (67, 64, 55, 48)


def test_chord_root_from_pitch_content():
    assert chord_root_pc((67, 64, 55, 48)) == 0  # C major
    assert chord_root_pc((67, 64, 60, 52)) == 0  # first inversion
    assert chord_root_pc((64, 60, 55, 48)) == 0
    assert chord_root_pc((72, 64, 60, 48)) == 0  # no 5th
    assert chord_root_pc((69, 64, 60, 57)) == 9  # A minor
    assert chord_root_pc((67, 65, 59, 43)) == 7  # G7 without its 5th
    assert chord_root_pc((67, 65, 62, 47)) == 7  # G7 complete, first inversion
    assert chord_root_pc((72, 65, 62, 57)) == 2  # D minor 7
    assert chord_root_pc((69, 65, 62, 59)) == 11  # B half-diminished 7
    assert chord_root_pc((65, 62, 59, 50)) == 11  # B diminished


def test_default_weight_formula():
    # No previous chord: every weight is 0.
    assert default_weights(state(None), [I_ROOT, (67, 62, 59, 43)]) == [0, 0]
    # I -> V: A moves 2, T moves 4, B moves 5. Root C -> G is a 5th.
    assert pair_weight(I_ROOT, (67, 62, 59, 43)) == 3 * 0 - (2 + 4 + 5) + 4
    # Same chord again: three held voices, root and bass unchanged.
    assert pair_weight(I_ROOT, I_ROOT) == 3 * 3 - 0 - 6
    # I -> I6: A and T held, bass up 4, same root but a different bass.
    assert pair_weight(I_ROOT, (67, 64, 55, 52)) == 3 * 2 - 4
    # I -> vi: root moves a 3rd, so no bonus. A held, T +2, B -3.
    assert pair_weight(I_ROOT, (69, 64, 57, 45)) == 3 * 1 - (2 + 3)
    # I -> IV: root up a 4th.
    assert pair_weight(I_ROOT, (69, 65, 57, 41)) == 0 - (1 + 2 + 7) + 4
    # The soprano never counts.
    assert pair_weight((60, 64, 55, 48), (84, 64, 55, 52)) == pair_weight(I_ROOT, (67, 64, 55, 52))
    options = [I_ROOT, (67, 62, 59, 43), (67, 64, 55, 52)]
    assert default_weights(state(I_ROOT), options) == [3, -7, 2]


def test_weights_are_pure_and_deterministic():
    options = [(67, 62, 59, 43), I_ROOT, (67, 64, 55, 52), (67, 60, 55, 52)]
    st = state(I_ROOT)
    frozen_options = copy.deepcopy(options)
    first = default_weights(st, options)
    assert options == frozen_options  # inputs untouched
    assert st == state(I_ROOT)
    for _ in range(3):
        assert default_weights(st, options) == first
    assert all(type(w) is int for w in first)
    # One weight per option, independent of the other options and their order.
    assert default_weights(st, options[::-1]) == first[::-1]
    assert [default_weights(st, [c])[0] for c in options] == first
    # Lists and tuples give the same result.
    assert default_weights(st, [list(c) for c in options]) == first
    # Only prev_chord matters to the phase 1 default.
    other = state(I_ROOT, ppb=50, chorale_id=99, phrase_index=4, beat_index=7,
                  key=Key("A", "minor"), soprano_midi=60)
    assert default_weights(other, options) == first


def test_weight_state_is_frozen():
    st = state(I_ROOT)
    with pytest.raises(Exception):
        st.prev_chord = None
    assert hash(st) == hash(state(I_ROOT))
    assert [f for f in st.__dataclass_fields__] == [
        "chorale_id", "phrase_index", "beat_index", "key", "soprano_midi",
        "prev_chord", "prev_prev_bass",
    ]


def test_custom_weights_must_return_one_int_per_option():
    options = [I_ROOT, (67, 62, 59, 43)]
    assert checked_weights(lambda s, o: [5, 7], state(None), options) == [5, 7]
    with pytest.raises(ValueError):
        checked_weights(lambda s, o: [1], state(None), options)
    with pytest.raises(TypeError):
        checked_weights(lambda s, o: [1.0, 2.0], state(None), options)
