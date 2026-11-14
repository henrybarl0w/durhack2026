"""Milestone 1: corpus loading, filtering, beats, phrases and fermatas."""

from fractions import Fraction

import music21
import pytest

from chorale_engine import ChoraleRejected, load_tune, rejected_chorales, usable_chorales
from chorale_engine import corpus


def beat_rows(tune, lo, hi):
    return [
        (b.offset, b.soprano_midi, b.phrase_index, b.beat_index, b.is_cadence, b.length)
        for b in tune.beats
        if lo <= b.offset < hi
    ]


def test_riemenschneider_numbering_comes_from_music21():
    ids = corpus.chorale_ids()
    assert ids[0] == 1 and ids[-1] == 371 and len(ids) == 371
    assert corpus.load_raw(2)["title"].startswith("Ich dank")
    assert load_tune(16).title.startswith("Es woll")


def test_music21_version_is_the_pinned_one():
    import tomllib
    from pathlib import Path

    data = tomllib.loads((Path(__file__).parent.parent / "pyproject.toml").read_text())
    assert f"music21=={music21.__version__}" in data["project"]["dependencies"]


def test_anacrusis_beats_start_at_offset_zero():
    # Chorale 16 has a one-crotchet upbeat: the first bar is one beat long.
    tune = load_tune(16)
    assert tune.measures[0].length == 1
    assert tune.measures[1].offset == 1 and tune.measures[1].length == 4
    assert beat_rows(tune, 0, 6) == [
        (0, 73, 0, 0, False, 1),
        (1, 74, 0, 1, False, 1),
        (2, 73, 0, 2, False, 1),
        (3, 71, 0, 3, False, 1),
        (4, 73, 0, 4, False, 1),
        (5, 76, 0, 5, False, 1),
    ]
    assert tune.phrases[0] == (0, 10)
    assert tune.bach_opening == (73, 66, 58, 54)


def test_passing_quavers_are_not_beats():
    # Chorale 16: C#5 quaver at 4, D5 quaver at 4.5, then E5 at 5.
    tune = load_tune(16)
    written = [(e.offset, e.midi) for e in tune.soprano if 4 <= e.offset < 6]
    assert written == [(4, 73), (Fraction(9, 2), 74), (5, 76)]
    assert [(b.offset, b.soprano_midi) for b in tune.beats if 4 <= b.offset < 6] == [
        (4, 73),
        (5, 76),
    ]
    # Chorale 14 has semiquavers off the beat (9.5 and 9.75).
    tune = load_tune(14)
    written = [(e.offset, e.midi) for e in tune.soprano if 9 <= e.offset < 11]
    assert written == [(9, 78), (Fraction(19, 2), 79), (Fraction(39, 4), 81), (10, 79)]
    assert [(b.offset, b.soprano_midi) for b in tune.beats if 9 <= b.offset < 11] == [
        (9, 78),
        (10, 79),
    ]
    assert all(b.offset == int(b.offset) for b in tune.beats)


def test_held_fermata_is_one_beat_and_rest_has_no_beat():
    # Chorale 16: fermata minim at 9 (two beats), crotchet rest at 11, new phrase at 12.
    tune = load_tune(16)
    assert beat_rows(tune, 8, 13) == [
        (8, 74, 0, 8, False, 1),
        (9, 73, 0, 9, True, 2),
        (12, 76, 1, 0, False, 1),
    ]
    # Fermata dotted minim at 21 lasts three beats; the next beat is 24.
    assert beat_rows(tune, 20, 25) == [
        (20, 67, 1, 8, False, 1),
        (21, 66, 1, 9, True, 3),
        (24, 69, 2, 0, False, 1),
    ]


def test_tied_notes_can_carry_more_than_one_chord():
    # Chorale 49: A4 crotchet tied to A4 crotchet, no fermata. Two beats, two decisions.
    tune = load_tune(49)
    tied = [(e.offset, e.length, e.midi, e.tie) for e in tune.soprano if 24 <= e.offset < 26]
    assert tied == [(24, 1, 69, "start"), (25, 1, 69, "stop")]
    assert beat_rows(tune, 23, 27) == [
        (23, 62, 2, 7, True, 1),
        (24, 69, 3, 0, False, 1),
        (25, 69, 3, 1, False, 1),
        (26, 67, 3, 2, False, 1),
    ]


def test_tie_into_a_fermata_note_keeps_the_earlier_beats():
    # Chorale 16 ends with F#4 for 2 beats tied to F#4 for 3 beats, fermata on the second.
    # The first two beats are ordinary decisions. The cadence is where the fermata is.
    tune = load_tune(16)
    tied = [(e.offset, e.length, e.midi, e.tie, e.fermata) for e in tune.soprano if e.tie]
    assert tied == [(63, 2, 66, "start", False), (65, 3, 66, "stop", True)]
    assert beat_rows(tune, 62, 70) == [
        (62, 67, 6, 6, False, 1),
        (63, 66, 6, 7, False, 1),
        (64, 66, 6, 8, False, 1),
        (65, 66, 6, 9, True, 3),
    ]


def test_tie_out_of_a_fermata_note_is_part_of_the_held_cadence():
    # Chorale 227 ends with a fermata D5 for 3 beats tied to one more beat, then rests.
    tune = load_tune(227)
    tail = [(e.offset, e.length, e.midi, e.tie, e.fermata) for e in tune.soprano[-4:]]
    assert tail == [
        (73, 3, 74, "start", True),
        (76, 1, 74, "stop", False),
        (77, 1, None, None, False),
        (78, 2, None, None, False),
    ]
    last = tune.beats[-1]
    assert (last.offset, last.soprano_midi, last.is_cadence, last.length) == (73, 74, True, 4)


def test_phrases_end_at_cadences_and_cover_all_beats():
    for cid in (6, 14, 16, 47, 227):
        tune = load_tune(cid)
        assert tune.phrases[0][0] == 0
        assert tune.phrases[-1][1] == len(tune.beats)
        for p, (start, end) in enumerate(tune.phrases):
            beats = tune.phrase_beats(p)
            assert [b.is_cadence for b in beats] == [False] * (end - start - 1) + [True]
            assert [b.phrase_index for b in beats] == [p] * (end - start)
            assert [b.beat_index for b in beats] == list(range(end - start))
            assert tune.beat(p, 0) is beats[0]
        for a, b in zip(tune.phrases, tune.phrases[1:]):
            assert a[1] == b[0]
        assert [b.index for b in tune.beats] == list(range(len(tune.beats)))


@pytest.mark.parametrize("cid", [6, 14, 16, 47, 49, 227])
def test_beats_match_an_independent_reading_of_the_score(cid):
    """Recompute the soprano pitch at every beat straight from music21."""
    tune = load_tune(cid)
    sop = corpus.parse_score(cid).parts[0].flatten()
    notes = [
        (Fraction(n.offset), Fraction(n.offset) + Fraction(n.duration.quarterLength), n)
        for n in sop.notesAndRests
        if n.duration.quarterLength > 0
    ]
    for b in tune.beats:
        sounding = [n for lo, hi, n in notes if lo <= b.offset < hi]
        assert len(sounding) == 1 and sounding[0].isNote
        assert sounding[0].pitch.midi == b.soprano_midi
    # Every integer offset that is not a beat is a rest or inside a held cadence note.
    beat_offsets = {b.offset for b in tune.beats}
    cadence_spans = [(b.offset, b.offset + b.length) for b in tune.beats if b.is_cadence]
    end = max(hi for _lo, hi, _n in notes)
    for t in range(int(end)):
        if t in beat_offsets:
            continue
        sounding = [n for lo, hi, n in notes if lo <= t < hi]
        resting = not sounding or sounding[0].isRest
        held = any(lo < t < hi for lo, hi in cadence_spans)
        assert resting or held, f"offset {t} should be a beat"


def test_filter_reasons():
    rejected = rejected_chorales()
    usable = usable_chorales()
    assert usable == sorted(usable)
    assert set(usable).isdisjoint(rejected)
    assert sorted(set(usable) | set(rejected)) == list(corpus.chorale_ids())
    assert len(usable) >= 20
    # Chorale 1 is in 3/4. Only 4/4 is accepted.
    assert rejected[1].startswith("meter not 4/4")
    with pytest.raises(ChoraleRejected) as err:
        load_tune(1)
    assert "meter not 4/4" in str(err.value)
    with pytest.raises(ChoraleRejected):
        load_tune(9999)
    for cid in usable:
        raw = corpus.load_raw(cid)
        assert raw["n_parts"] == 4 and raw["meters"] == ["4/4"]
        assert any(e["fermata"] for e in raw["soprano"])


def test_deferred_groups_are_rejected_for_now():
    rejected = rejected_chorales()
    # Chorale 2 (A major) has a G natural, so its whole key map is B minor.
    assert rejected[2] == "the key map never uses the home key"
    # Chorale 3 (A minor) ends on B, outside the A minor triad.
    assert rejected[3] == "the final soprano note is not in the home tonic triad"
    for cid in usable_chorales():
        tune = load_tune(cid)
        assert any(k == tune.home_key for k in tune.key_map)
        tonic = tune.home_key.tonic_pc
        third = 3 if tune.home_key.mode == "minor" else 4
        allowed = {tonic, (tonic + third) % 12, (tonic + 4) % 12, (tonic + 7) % 12}
        assert tune.beats[-1].soprano_midi % 12 in allowed


def test_cache_matches_fresh_extraction():
    for cid in (14, 16):
        cached = corpus.load_raw(cid)
        assert corpus.extract_raw(cid) == cached
