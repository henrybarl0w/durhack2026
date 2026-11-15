"""Experimental: Bach-trained weights and off-beat ornaments."""

import json
import random

import pytest

from chorale_engine import capacity_report, decode, decode_musicxml, encode, load_tune, read_musicxml, write_musicxml
from chorale_engine.bach_weights import MODEL_PATH, bach_weights, chord_class
from chorale_engine.checker import check_musicxml
from chorale_engine.keys import Key
from chorale_engine.musicxml_io import read_notes
from chorale_engine.ornaments import ornaments
from chorale_engine.weights import WeightState

SAMPLE = [6, 16, 47, 58, 215]


def test_model_file_is_all_ints():
    def walk(node):
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        else:
            assert type(node) is int
    walk(json.loads(MODEL_PATH.read_text()))


def test_bach_weights_are_pure_ints():
    state = WeightState(6, 0, 1, Key("C", "major"), 67, (67, 64, 55, 48), None)
    options = [(67, 62, 59, 43), (67, 64, 55, 48), (67, 64, 60, 52), (67, 62, 55, 47)]
    first = bach_weights(state, options)
    assert all(type(w) is int for w in first) and len(first) == 4
    assert bach_weights(state, options) == first
    assert bach_weights(state, options[::-1]) == first[::-1]
    # Moving smoothly (the bass a step down to V6) beats standing completely still,
    # and standing still beats a leap of a 4th in the bass.
    assert first[3] == max(first)
    assert first[3] > first[1] > first[0]
    # With no previous chord it still gives one int per option.
    assert all(type(w) is int for w in bach_weights(WeightState(6, 0, 0, Key("C", "major"), 67, None, None), options))
    assert chord_class((67, 64, 55, 48), 0) == "0.470.0"  # C major, root position, in C


@pytest.mark.parametrize("cid", SAMPLE)
def test_bach_weights_round_trip(cid):
    cap = capacity_report(cid, bach_weights)["capacity"]
    rng = random.Random(cid)
    for n in (0, cap - 1, rng.randrange(cap)):
        assert decode(cid, encode(cid, n, bach_weights), bach_weights) == n


@pytest.mark.parametrize("cid", SAMPLE)
def test_ornaments_do_not_change_the_decoded_number(cid, tmp_path):
    tune = load_tune(cid)
    cap = capacity_report(cid, bach_weights)["capacity"]
    n = random.Random(cid + 1).randrange(cap)
    chords = encode(cid, n, bach_weights)
    extra = ornaments(cid, chords)
    assert extra == ornaments(cid, chords)  # deterministic
    assert len(extra) == len(chords) and any(any(row) for row in extra)
    assert ornaments(cid, chords, walking=False) != extra
    assert all(sum(x is not None for x in row) <= 2 for row in extra)  # the bass and one inner voice at most
    assert sum(1 for row in extra if row[2] is not None) > 0.4 * len(extra)  # the bass walks
    plain, fancy = tmp_path / "plain.musicxml", tmp_path / "fancy.musicxml"
    write_musicxml(cid, chords, plain)
    write_musicxml(cid, chords, fancy, ornamented=True)
    assert plain.read_bytes() != fancy.read_bytes()
    assert read_musicxml(cid, fancy) == chords
    assert decode(cid, read_musicxml(cid, fancy), bach_weights) == n
    assert check_musicxml(cid, fancy) == []
    assert decode_musicxml(fancy, bach_weights)[1] == n
    onsets = {e.offset for e in tune.soprano if e.midi is not None}
    for i, row in enumerate(extra):
        if not any(row):
            continue
        beat = tune.beats[i]
        assert not beat.is_cadence
        off = list(chords[i])
        for v, pitch in zip((1, 2, 3), row):
            if pitch is not None:
                off[v] = pitch
                assert pitch % 12 in beat.key.pitch_classes()
        assert off[1] >= off[2] >= off[3]
        soprano_moves = any(beat.offset < o < beat.offset + 1 for o in onsets)
        for v, pitch in zip((1, 2), row[:2]):
            if pitch is None:
                continue
            # Inner voices move only by step, and only while the soprano holds its note.
            assert not soprano_moves
            assert abs(pitch - chords[i][v]) in (1, 2) and abs(chords[i + 1][v] - pitch) in (1, 2)
        if row[2] is not None:
            # The walking bass stays in range and never leaps more than a 5th, or an octave.
            assert 40 <= row[2] <= 60
            lead = abs(row[2] - chords[i][3])
            assert lead <= 7 or lead == 12
    # The file really has notes on the half beats in the lower voices.
    half = [n for n in read_notes(fancy) if n[2] in ("2", "3", "4") and n[3].denominator == 2]
    assert len(half) == sum(1 for row in extra for x in row if x is not None)


def test_default_weights_are_untouched(tmp_path):
    # The experimental pieces are opt-in: the default encoding and file are as before.
    chords = encode(6, 4242)
    path = tmp_path / "s.musicxml"
    write_musicxml(6, chords, path)
    assert decode_musicxml(path) == (6, 4242)
    assert chords != encode(6, 4242, bach_weights)


@pytest.mark.parametrize("cid", SAMPLE)
def test_ornament_signs_do_not_change_the_decoded_number(cid, tmp_path):
    import xml.etree.ElementTree as ET

    import music21

    from chorale_engine.ornaments import ornament_signs, play_sign

    signs = ornament_signs(cid)
    assert signs == ornament_signs(cid) and signs
    tune = load_tune(cid)
    for index, name in signs.items():
        e = tune.soprano[index]
        assert e.midi is not None and e.offset.denominator == 1 and e.tie in (None, "start")
        assert name in ("trill-mark", "turn", "mordent", "inverted-mordent")
        played = play_sign(name, tune.home_key, e.midi, e.length if e.length >= 1 else 1)
        assert sum(d for d, _ in played) == (e.length if e.length >= 1 else 1)
    n = random.Random(cid + 2).randrange(capacity_report(cid, bach_weights)["capacity"])
    chords = encode(cid, n, bach_weights)
    path = tmp_path / "s.musicxml"
    write_musicxml(cid, chords, path, ornamented=True, signs=True)
    written = [el.tag for el in ET.parse(path).getroot().iter() if el.tag in ("trill-mark", "turn", "mordent", "inverted-mordent")]
    assert sorted(written) == sorted(signs.values())
    assert read_musicxml(cid, path) == chords
    assert decode_musicxml(path, bach_weights)[1] == n
    assert check_musicxml(cid, path) == []
    # music21 reads the signs as ornaments on the notes.
    found = [x for n in music21.converter.parse(str(path)).recurse().notes for x in n.expressions
             if isinstance(x, (music21.expressions.Trill, music21.expressions.Turn, music21.expressions.GeneralMordent))]
    assert len(found) == len(signs)


def test_strict_style_round_trip_and_rules():
    from chorale_engine.bach_weights import chord_class, load_model
    from chorale_engine.chords import analyse, beat_context
    from chorale_engine.style import ALLOWED_BASS_LEAPS, good_doubling, strict_weights

    after = load_model()["chord"]["after"]
    rng = random.Random(77)
    bad_doubling = total = 0
    for cid in (6, 16, 47):
        tune = load_tune(cid)
        cap = capacity_report(cid, strict_weights)["capacity"]
        for n in (0, cap - 1, rng.randrange(cap)):
            chords = encode(cid, n, strict_weights)
            assert decode(cid, chords, strict_weights) == n
            for i in range(1, len(chords)):
                a, b = chords[i - 1], chords[i]
                if tune.beats[i].phrase_index == tune.beats[i - 1].phrase_index:
                    assert abs(b[3] - a[3]) in ALLOWED_BASS_LEAPS, (cid, i)
                    tonic = tune.beats[i].key.tonic_pc
                    row = after.get(chord_class(a, tonic))
                    if row is not None:
                        assert row.get(chord_class(b, tonic), -10 ** 9) >= -581, (cid, i)  # 0.3% floor
            for i, c in enumerate(chords):
                total += 1
                bad_doubling += not good_doubling(analyse(beat_context(tune, i), c))
    # Doubling is a strong preference, not a hard rule: it may slip only where no better option exists.
    assert bad_doubling < 0.2 * total
