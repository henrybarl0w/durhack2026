"""Milestone 2: chord vocabulary and voicing generation."""

import random

from chorale_engine import load_tune, usable_chorales
from chorale_engine.chords import (
    A_RANGE,
    B_RANGE,
    T_RANGE,
    BeatContext,
    analyse,
    beat_context,
    brute_force_voicings,
    canonical_key,
    incremental_voicings,
    is_home_tonic,
    spell,
    vocabulary,
)
from chorale_engine.keys import Key
from chorale_engine.rules import legal_candidates, violations

C = Key("C", "major")
AM = Key("A", "minor")


def ctx(key, soprano, home=None, **kw):
    return BeatContext(key, soprano, home or key, **kw)


def names(c):
    return [t.name for t in vocabulary(c)]


def test_major_vocabulary():
    assert names(ctx(C, 67)) == ["I", "ii", "iii", "IV", "V", "vi", "vii°", "V7", "ii7", "viiø7"]
    by_name = {t.name: t for t in vocabulary(ctx(C, 67))}
    assert by_name["I"].pcs == (0, 4, 7)
    assert by_name["vii°"].pcs == (11, 2, 5)
    assert by_name["V7"].pcs == (7, 11, 2, 5)
    assert by_name["ii7"].pcs == (2, 5, 9, 0)
    assert by_name["viiø7"].pcs == (11, 2, 5, 9)
    assert sorted(n for n, t in by_name.items() if t.dominant) == ["V", "V7", "vii°", "viiø7"]


def test_minor_vocabulary_is_harmonic_minor():
    by_name = {t.name: t for t in vocabulary(ctx(AM, 69))}
    assert list(by_name) == ["i", "ii°", "III", "iv", "V", "VI", "vii°", "V7", "iiø7", "vii°7"]
    assert by_name["i"].pcs == (9, 0, 4)
    assert by_name["III"].pcs == (0, 4, 7)  # major, with the natural 7th, not augmented
    assert by_name["III"].quality == "major"
    assert by_name["V"].pcs == (4, 8, 11)  # raised 7th
    assert by_name["vii°7"].pcs == (8, 11, 2, 5)
    assert by_name["iiø7"].pcs == (11, 2, 5, 9)


def test_raised_sixth_chords_only_under_a_raised_sixth_soprano():
    # F#4 is the raised 6th of A minor.
    with_f_sharp = {t.name: t for t in vocabulary(ctx(AM, 66))}
    assert with_f_sharp["IV"].pcs == (2, 6, 9) and with_f_sharp["IV"].quality == "major"
    assert with_f_sharp["ii"].pcs == (11, 2, 6) and with_f_sharp["ii"].quality == "minor"
    for soprano in (60, 62, 64, 65, 67, 68, 69, 71):
        assert "IV" not in names(ctx(AM, soprano)) and "ii" not in names(ctx(AM, soprano))
    # Every voicing under the raised 6th is IV or ii, since no other chord holds it.
    kinds = {analyse(ctx(AM, 66), v).ctype.name for v in brute_force_voicings(ctx(AM, 66))}
    assert kinds == {"IV", "ii"}


def test_diminished_triads_only_in_first_inversion():
    for c in (ctx(C, 62), ctx(C, 65), ctx(C, 71), ctx(AM, 62), ctx(AM, 65), ctx(AM, 71), ctx(AM, 68)):
        seen = set()
        for v in brute_force_voicings(c):
            info = analyse(c, v)
            if not info.ctype.is_seventh and info.ctype.quality == "diminished":
                seen.add(info.ctype.name)
                assert info.inversion == 1, (v, info.ctype.name)
                assert v[3] % 12 == info.ctype.third_pc
        assert seen, "expected some diminished triads"
    # Seventh chords keep all four inversions.
    c = ctx(C, 67)
    inversions = {
        analyse(c, v).inversion for v in brute_force_voicings(c) if analyse(c, v).ctype.name == "V7"
    }
    assert inversions == {0, 1, 2, 3}
    # A root-position diminished triad is not a voicing: B D F with B in the bass.
    assert analyse(ctx(C, 65), (65, 62, 59, 47)) is None
    assert analyse(ctx(C, 65), (65, 62, 59, 50)).ctype.name == "vii°"


def test_picardy_third_at_the_final_beat_only():
    # C#5 over A minor home: only a voicing at the final beat.
    chord = (73, 64, 57, 45)
    final = ctx(AM, 73, is_final=True)
    assert "home I (Picardy)" in names(final)
    info = analyse(final, chord)
    assert info is not None and info.ctype.pcs == (9, 1, 4)
    assert is_home_tonic(final, info.ctype, picardy=True)
    assert not is_home_tonic(final, info.ctype, picardy=False)
    for other in (ctx(AM, 73), ctx(AM, 73, is_opening=True)):
        assert "home I (Picardy)" not in names(other)
        assert analyse(other, chord) is None
        assert brute_force_voicings(other) == ()
    # With E in the soprano the Picardy chord is offered next to the minor tonic.
    final_e = ctx(AM, 76, is_final=True)
    thirds = {
        m % 12
        for v in brute_force_voicings(final_e)
        if is_home_tonic(final_e, analyse(final_e, v).ctype)
        for m in v
        if m % 12 in (0, 1)
    }
    assert thirds == {0, 1}  # both C natural and C sharp voicings exist
    mid = ctx(AM, 76)
    assert all(1 not in {m % 12 for m in v} for v in brute_force_voicings(mid))
    # A major home key has no Picardy chord.
    assert names(ctx(C, 72, is_final=True)) == names(ctx(C, 72))


def test_home_tonic_is_added_at_the_opening_and_final_beat_of_a_foreign_key():
    # Home A major, beat in B minor: the A major triad is not in B minor's vocabulary.
    a_major = Key("A", "major")
    b_minor = Key("B", "minor")
    chord = (69, 64, 61, 57)
    assert analyse(BeatContext(b_minor, 69, a_major), chord) is None
    for kw in ({"is_opening": True}, {"is_final": True}):
        c = BeatContext(b_minor, 69, a_major, **kw)
        info = analyse(c, chord)
        assert info.ctype.name == "home I" and info.inversion == 0
        assert is_home_tonic(c, info.ctype)
    # When the home tonic is already a chord of the beat's key it is not added twice.
    c = BeatContext(Key("G", "major"), 72, C, is_final=True)
    assert names(c) == names(BeatContext(Key("G", "major"), 72, C))
    assert is_home_tonic(c, analyse(c, (72, 67, 64, 48)).ctype)


def test_voicings_respect_ranges_and_required_tones():
    for c in (ctx(C, 67), ctx(C, 72), ctx(AM, 69), ctx(AM, 66), ctx(AM, 76, is_final=True)):
        voicings = brute_force_voicings(c)
        assert voicings and list(voicings) == sorted(set(voicings), key=canonical_key)
        for v in voicings:
            s, a, t, b = v
            assert s == c.soprano_midi
            assert A_RANGE[0] <= a <= A_RANGE[1]
            assert T_RANGE[0] <= t <= T_RANGE[1]
            assert B_RANGE[0] <= b <= B_RANGE[1]
            info = analyse(c, v)
            pcs = {m % 12 for m in v}
            assert pcs <= set(info.ctype.pcs)
            assert info.ctype.pcs[0] in pcs and info.ctype.pcs[1] in pcs
            if info.ctype.is_seventh:
                assert info.ctype.pcs[3] in pcs
    c = ctx(C, 67)
    assert analyse(c, (67, 62, 55, 43)) is None  # G D G G: no 3rd
    assert analyse(c, (67, 59, 55, 43)).ctype.name == "V"  # 5th omitted
    assert analyse(c, (67, 62, 59, 43)).ctype.name == "V"
    assert analyse(c, (67, 65, 62, 43)) is None  # G F D G: V7 without its 3rd
    assert analyse(c, (67, 65, 59, 43)).ctype.name == "V7"  # 5th omitted
    assert analyse(c, (67, 75, 59, 43)) is None  # alto out of range
    assert analyse(c, (72, 64, 55, 48)) is None  # wrong soprano for this beat


def test_spelling_follows_the_key():
    assert spell(ctx(AM, 68), (68, 64, 59, 52)) == (("G", 1, 4), ("E", 0, 4), ("B", 0, 3), ("E", 0, 3))
    e_flat = Key("E-", "major")
    assert spell(ctx(e_flat, 70), (70, 67, 63, 51)) == (
        ("B", -1, 4), ("G", 0, 4), ("E", -1, 4), ("E", -1, 3),
    )
    # C flat and B sharp land in the right octave.
    g_flat = Key("G-", "major")
    assert spell(ctx(g_flat, 71), (71, 68, 63, 56))[0] == ("C", -1, 5)
    c_sharp_minor = Key("C#", "minor")
    assert spell(ctx(c_sharp_minor, 72), (72, 68, 63, 56))[0] == ("B", 1, 4)


def test_incremental_voicings_are_a_window_of_all_voicings():
    c = ctx(C, 67)
    prev = (67, 64, 55, 48)
    window = incremental_voicings(c, prev)
    everything = brute_force_voicings(c)
    expected = [v for v in everything if abs(v[1] - prev[1]) <= 5 and abs(v[2] - prev[2]) <= 5]
    assert window == expected and 0 < len(window) < len(everything)
    assert incremental_voicings(c, None) == list(everything)


def _sample_state(rng, tune, t):
    """A legal (prev_chord, prev_prev_bass) for beat t, or None if the draw fails."""
    prev_ctx = beat_context(tune, t - 1)
    if t == 1:
        cands = legal_candidates(prev_ctx)
        return (rng.choice(cands), None) if cands else None
    pp_ctx = beat_context(tune, t - 2)
    firsts = legal_candidates(pp_ctx)
    if not firsts:
        return None
    pp = rng.choice(firsts)
    cands = legal_candidates(prev_ctx, pp_ctx, pp, None)
    if not cands:
        return None
    return rng.choice(cands), pp[3]


def test_incremental_equals_brute_force_on_sampled_beats():
    rng = random.Random(20260)
    chorales = rng.sample(usable_chorales(), 8)
    checked = 0
    chorales_checked = set()
    nonempty = 0
    for cid in chorales:
        tune = load_tune(cid)
        done = 0
        attempts = 0
        while done < 5 and attempts < 60:
            attempts += 1
            t = rng.randrange(1, len(tune.beats))
            state = _sample_state(rng, tune, t)
            if state is None:
                continue
            prev, ppb = state
            c, prev_ctx = beat_context(tune, t), beat_context(tune, t - 1)
            fast = legal_candidates(c, prev_ctx, prev, ppb, incremental=True)
            slow = legal_candidates(c, prev_ctx, prev, ppb, incremental=False)
            assert fast == slow, (cid, t, prev, ppb)
            # Brute force really is every voicing, filtered by the rules.
            by_hand = [
                v for v in brute_force_voicings(c)
                if not violations(c, v, prev_ctx, prev, ppb)
            ]
            assert slow == by_hand
            nonempty += bool(slow)
            done += 1
            checked += 1
            chorales_checked.add(cid)
    assert checked >= 20 and len(chorales_checked) >= 5
    assert nonempty >= 10
    # With no previous chord both generators give every statically legal voicing.
    c = beat_context(load_tune(chorales[0]), 0)
    assert legal_candidates(c, incremental=True) == legal_candidates(c, incremental=False)
