"""Milestone 1: home key, candidate keys and the key map."""

import itertools
import random

import pytest

from chorale_engine import load_tune, usable_chorales
from chorale_engine import corpus
from chorale_engine.keys import Key, NoKeyMap, candidate_keys, key_map


def test_pitch_class_sets():
    assert Key("C", "major").pitch_classes() == frozenset({0, 2, 4, 5, 7, 9, 11})
    # A minor: natural minor plus F# and G#.
    assert Key("A", "minor").pitch_classes() == frozenset({9, 11, 0, 2, 4, 5, 7, 6, 8})
    assert Key("B-", "major").tonic_pc == 10
    assert Key("F#", "minor").tonic_pc == 6


def test_candidate_key_order():
    assert [str(k) for k in candidate_keys(Key("C", "major"))] == [
        "C major", "G major", "F major", "A minor", "E minor", "D minor",
    ]
    assert [str(k) for k in candidate_keys(Key("A", "minor"))] == [
        "A minor", "E minor", "D minor", "C major", "G major", "F major",
    ]
    assert [str(k) for k in candidate_keys(Key("B-", "major"))] == [
        "B- major", "F major", "E- major", "G minor", "D minor", "C minor",
    ]


@pytest.mark.parametrize("cid", [6, 14, 16, 47, 49, 227])
def test_home_key_equals_music21_full_score_analysis(cid):
    score = corpus.parse_score(cid)
    k = score.analyze("key")
    home = load_tune(cid).home_key
    assert (home.tonic, home.mode) == (k.tonic.name, k.mode)
    assert home.tonic_pc == k.tonic.pitchClass


def test_every_soprano_pitch_class_is_in_its_beats_key():
    usable = usable_chorales()
    assert len(usable) >= 20
    for cid in usable:
        tune = load_tune(cid)
        cands = candidate_keys(tune.home_key)
        assert len(tune.key_map) == len(tune.beats)
        for b, k in zip(tune.beats, tune.key_map):
            assert b.key == k
            assert k in cands
            assert b.soprano_midi % 12 in k.pitch_classes(), (cid, b.index)


def test_key_map_is_deterministic():
    for cid in (6, 15, 16, 47, 59):
        tune = load_tune(cid)
        pcs = [b.soprano_midi % 12 for b in tune.beats]
        first = key_map(pcs, tune.home_key)
        assert first == key_map(list(pcs), tune.home_key)
        assert tuple(first) == tune.key_map
        # Rebuilding the tune from the raw data gives the same map.
        rebuilt, reason = corpus._build(corpus.load_raw(cid))
        assert reason is None and rebuilt.key_map == tune.key_map


def _cost(assign):
    changes = sum(1 for a, b in zip(assign, assign[1:]) if a != b)
    return (changes, sum(1 for k in assign if k != 0))


def _reference_key_map(pcs, home):
    """Brute force over all assignments, with the left-to-right tie-break."""
    cands = candidate_keys(home)
    sets = [k.pitch_classes() for k in cands]
    legal = [
        a for a in itertools.product(range(6), repeat=len(pcs))
        if all(pc in sets[k] for pc, k in zip(pcs, a))
    ]
    if not legal:
        return None
    best = min(_cost(a) for a in legal)
    optimal = [a for a in legal if _cost(a) == best]
    chosen = []
    for t in range(len(pcs)):
        allowed = sorted({a[t] for a in optimal if list(a[:t]) == chosen})
        pick = chosen[-1] if chosen and chosen[-1] in allowed else allowed[0]
        chosen.append(pick)
    return [cands[k] for k in chosen]


def test_key_map_matches_brute_force():
    rng = random.Random(1)
    homes = [Key("C", "major"), Key("A", "minor"), Key("E-", "major"), Key("F#", "minor")]
    checked = 0
    for _ in range(300):
        home = rng.choice(homes)
        pcs = [rng.randrange(12) for _ in range(rng.randint(1, 6))]
        expected = _reference_key_map(pcs, home)
        if expected is None:
            with pytest.raises(NoKeyMap):
                key_map(pcs, home)
        else:
            assert key_map(pcs, home) == expected, (home, pcs)
            checked += 1
    assert checked > 100


def test_key_map_hand_cases():
    c = Key("C", "major")
    # All in C major: stay home.
    assert [str(k) for k in key_map([0, 4, 7, 0], c)] == ["C major"] * 4
    # F# then F natural: both are in A minor, so one key with no change wins
    # over fewer non-home beats with a change.
    assert [str(k) for k in key_map([0, 6, 5, 0], c)] == ["A minor"] * 4
    # B flat and G# share no candidate key, so one change is forced. F major is
    # the first candidate holding B flat, and it is kept for as long as possible.
    assert [str(k) for k in key_map([0, 10, 0, 8], c)] == [
        "F major", "F major", "F major", "A minor",
    ]
    assert key_map([], c) == []
    # The six candidate keys together hold all 12 pitch classes, so a key map
    # always exists.
    for home in (c, Key("A", "minor"), Key("E-", "major"), Key("F#", "minor")):
        covered = frozenset().union(*(k.pitch_classes() for k in candidate_keys(home)))
        assert covered == frozenset(range(12))
    # C and C# are both in E minor (natural and raised 6th), the first such candidate.
    assert [str(k) for k in key_map([0, 1], c)] == ["E minor", "E minor"]
