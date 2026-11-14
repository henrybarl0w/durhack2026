"""Milestone 2: hard rules 1-11, one passing and one failing case each.

All cases are in C major or A minor. Chords are (S, A, T, B) MIDI tuples.
"""

import pytest

from chorale_engine.chords import BeatContext, analyse
from chorale_engine.keys import Key
from chorale_engine.rules import is_legal, legal_candidates, static_violations, violations

C = Key("C", "major")
AM = Key("A", "minor")


def check(key, chord, prev=None, ppb=None):
    c = BeatContext(key, chord[0], key)
    prev_ctx = BeatContext(key, prev[0], key) if prev else None
    return violations(c, chord, prev_ctx, prev, ppb)


I_ROOT = (67, 64, 55, 48)  # G4 E4 G3 C3


def test_rule_1_parallel_fifths_and_octaves():
    # I -> ii with every voice up a tone: tenor and bass stay a 5th apart.
    assert 1 in check(C, (69, 65, 57, 50), I_ROOT)
    # I -> V with the soprano held: no parallels.
    assert check(C, (67, 62, 59, 43), I_ROOT) == frozenset()
    # Parallel octaves between soprano and bass: C-C to D-D.
    assert 1 in check(C, (74, 65, 57, 50), (72, 64, 55, 48))
    # Contrary motion from a 5th to a compound 5th still counts.
    assert 1 in check(C, (69, 65, 62, 50), (67, 64, 60, 60))
    # A held 5th is fine: neither voice moves.
    assert 1 not in check(C, (72, 64, 55, 48), I_ROOT)


def test_rule_2_hidden_fifths_and_octaves():
    # Soprano leaps up a 3rd, bass rises too, and they land on an octave.
    assert check(C, (67, 62, 59, 55), (64, 60, 55, 48)) == frozenset({2})
    # Soprano moves by step into the octave: allowed.
    assert 2 not in check(C, (67, 62, 59, 55), (65, 60, 57, 53))
    # Contrary motion into an octave: allowed.
    assert 2 not in check(C, (67, 62, 59, 43), (64, 60, 55, 48))


def test_rule_3_spacing():
    assert static_violations(BeatContext(C, 76, C), (76, 60, 55, 48)) == frozenset({3})  # S-A a 10th
    assert static_violations(BeatContext(C, 79, C), (79, 67, 52, 48)) == frozenset({3})  # A-T a 10th
    assert check(C, (76, 67, 60, 48)) == frozenset()
    # Tenor and bass may be more than an octave apart.
    assert check(C, (76, 72, 67, 48)) == frozenset()


def test_rule_4_crossing_and_overlap():
    assert check(C, (67, 60, 64, 48)) == frozenset({4})  # tenor above alto
    assert 4 not in check(C, (67, 64, 64, 48))  # unison allowed
    # Tenor moves to D4, above the alto's previous C4.
    assert 4 in check(C, (67, 62, 62, 47), (64, 60, 55, 48))
    # Soprano drops below the alto's previous pitch.
    assert 4 in check(C, (62, 59, 55, 43), (72, 64, 60, 48))
    assert 4 not in check(C, (67, 62, 59, 43), I_ROOT)


def test_rule_5_leading_tone():
    # iii with B doubled (soprano and tenor).
    assert check(C, (71, 67, 59, 52)) == frozenset({5})
    assert check(C, (71, 67, 64, 52)) == frozenset()
    # V6 (B in the bass) must rise to C.
    v6 = (67, 62, 55, 47)
    assert check(C, (67, 64, 55, 48), v6) == frozenset()
    assert 5 in check(C, (67, 60, 55, 52), v6)
    assert 5 in check(C, (67, 62, 55, 47), v6)  # staying put is not rising
    # The soprano's leading tone may fall: it is the given tune.
    assert 5 not in check(C, (67, 62, 59, 43), (71, 62, 55, 43))
    # An inner-voice leading tone is exempt too.
    assert 5 not in check(C, (67, 64, 55, 48), (67, 62, 59, 43))


def test_rule_6_seventh_resolves_down_by_step():
    v7 = (67, 65, 59, 43)  # F4 (the 7th) in the alto
    assert check(C, (67, 64, 60, 48), v7) == frozenset()  # F -> E
    assert 6 in check(C, (67, 67, 64, 48), v7)  # F -> G
    assert 6 in check(C, v7, v7)  # F held
    # 7th in the bass (V4/2) falls a semitone to I6.
    assert 6 not in check(C, (67, 60, 55, 52), (67, 62, 59, 53))
    assert 6 in check(C, (67, 64, 60, 48), (67, 62, 59, 53))
    # A 7th in the soprano is exempt.
    assert 6 not in check(C, (67, 64, 60, 48), (65, 62, 59, 43))


def test_rule_7_no_doubled_third_in_major_triads():
    assert check(C, (64, 64, 55, 48)) == frozenset({7})
    assert check(C, (64, 60, 55, 48)) == frozenset()
    # Minor triads may double the 3rd.
    assert check(C, (65, 65, 57, 50)) == frozenset()
    # III in minor is a major triad.
    assert 7 in check(AM, (76, 64, 55, 48))


def test_rule_8_inner_leaps():
    assert 8 in check(C, (72, 60, 52, 48), (72, 67, 64, 48))  # alto falls a 5th, tenor an octave
    assert 8 not in check(C, (72, 64, 60, 48), (72, 67, 64, 48))
    # A perfect 4th is the limit.
    assert 8 not in check(C, (72, 64, 60, 48), (72, 64, 55, 48))
    assert 8 in check(C, (72, 67, 64, 48), (72, 65, 57, 53))  # tenor up a 5th
    # The bass may leap further.
    assert 8 not in check(C, (67, 62, 59, 43), (67, 64, 60, 55))


def test_rule_9_second_inversion_triads():
    cadential = (72, 64, 60, 55)  # I 6/4
    assert analyse(BeatContext(C, 72, C), cadential).is_second_inversion_triad
    # Cadential: followed by root-position V over the same bass.
    assert check(C, (71, 62, 55, 55), cadential) == frozenset()
    assert check(C, (71, 65, 62, 55), cadential) == frozenset()  # V7
    assert 9 in check(C, (72, 64, 57, 57), cadential)  # goes to vi
    assert 9 in check(C, (71, 62, 55, 43), cadential)  # V, but the bass drops an octave
    # Passing: bass C - D - E.
    passing = (67, 59, 55, 50)  # V 6/4
    assert check(C, (67, 60, 55, 52), passing, 48) == frozenset()
    assert 9 in check(C, (67, 60, 55, 52), passing, None)
    assert 9 in check(C, (67, 60, 55, 52), passing, 55)  # approached by leap
    assert 9 in check(C, (67, 64, 55, 48), passing, 48)  # bass turns back
    # A 6/4 that is not the tonic must be approached by step.
    assert 9 in check(C, passing, (65, 60, 57, 53))
    assert 9 not in check(C, passing, I_ROOT)
    # With nothing before it only the tonic 6/4 is possible.
    assert 9 in check(C, passing)
    assert 9 not in check(C, cadential)
    # Second-inversion seventh chords are not restricted.
    assert 9 not in check(C, (67, 64, 60, 48), (67, 65, 59, 50))


def test_rule_10_augmented_second():
    # Alto F4 -> G#4 in A minor.
    assert 10 in check(AM, (71, 68, 64, 52), (69, 65, 62, 50))
    # And downwards, G#4 -> F4.
    assert 10 in check(AM, (69, 65, 62, 50), (71, 68, 64, 52))
    assert check(AM, (68, 64, 59, 52), (69, 64, 60, 45)) == frozenset()
    # The soprano may sing an augmented 2nd: it is the given tune.
    assert 10 not in check(AM, (68, 64, 59, 52), (65, 62, 57, 50))


def test_rule_11_cross_relation():
    iii = (72, 67, 64, 48)  # G natural in the alto
    # The alto itself moves G -> G#: allowed.
    assert check(AM, (71, 68, 64, 52), iii) == frozenset()
    # G natural in the alto, G# in the tenor: cross-relation.
    assert 11 in check(AM, (71, 64, 56, 52), iii)
    # Soprano G natural against a G# in the bass counts too (pairs include the soprano).
    assert 11 in check(AM, (76, 71, 64, 56), (67, 64, 60, 48))
    # Same notes in both chords: no cross-relation.
    assert 11 not in check(AM, (72, 67, 64, 48), iii)


def test_is_legal_and_disabled_rules():
    c = BeatContext(C, 69, C)
    prev_ctx = BeatContext(C, 67, C)
    assert not is_legal(c, (69, 65, 57, 50), prev_ctx, I_ROOT)
    assert is_legal(c, (69, 65, 57, 50), prev_ctx, I_ROOT, disabled=frozenset({1}))
    assert not is_legal(c, (69, 65, 57, 49), prev_ctx, I_ROOT)  # not a voicing at all
    with pytest.raises(ValueError):
        violations(c, (69, 65, 57, 49), prev_ctx, I_ROOT)
    strict = legal_candidates(c, prev_ctx, I_ROOT)
    loose = legal_candidates(c, prev_ctx, I_ROOT, disabled=frozenset({1}))
    assert set(strict) < set(loose)
    for incremental in (True, False):
        for off in (frozenset(), frozenset({8}), frozenset({1, 8})):
            assert legal_candidates(c, prev_ctx, I_ROOT, incremental=incremental, disabled=off) == \
                legal_candidates(c, prev_ctx, I_ROOT, incremental=not incremental, disabled=off)


def test_rules_are_deterministic():
    c = BeatContext(AM, 71, AM)
    prev_ctx = BeatContext(AM, 72, AM)
    first = legal_candidates(c, prev_ctx, (72, 67, 64, 48))
    assert first == legal_candidates(c, prev_ctx, (72, 67, 64, 48))
    assert first == sorted(first, key=lambda ch: (ch[3], ch[2], ch[1], ch[0]))
