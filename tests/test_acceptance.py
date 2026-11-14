"""Milestone 5 and acceptance: MusicXML, checker, CLI, determinism."""

import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import music21
import pytest

from chorale_engine import (
    CapacityError,
    ChordNotInOptions,
    capacity_report,
    decode,
    decode_phrase,
    encode,
    encode_phrase,
    load_tune,
    phrase_capacity,
    read_musicxml,
    usable_chorales,
    write_musicxml,
)
from chorale_engine.checker import check_musicxml
from chorale_engine.realise import realise

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run(
        [sys.executable, "-m", "chorale_engine", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_realise_is_the_identity():
    chords = encode(6, 7)
    assert realise(chords) == chords
    assert realise([list(c) for c in chords]) == chords


def test_musicxml_layout(tmp_path):
    cid = 16
    path = tmp_path / "s.musicxml"
    write_musicxml(cid, encode(cid, 99), path)
    root = ET.parse(path).getroot()
    assert root.find("work/work-title").text == "Chorale 16"
    assert [(p.get("id"), p.find("part-name").text) for p in root.find("part-list")] == [
        ("P1", "S"), ("P2", "A"), ("P3", "T"), ("P4", "B"),
    ]
    assert root.find("identification") is None  # no encoding date
    clefs = []
    for part in root.findall("part"):
        assert {v.text for v in part.iter("voice")} == {"1"}
        assert part.find("measure/attributes/staves").text == "1"
        assert not list(part.iter("chord")) and not list(part.iter("backup"))
        clef = part.find("measure/attributes/clef")
        change = clef.find("clef-octave-change")
        clefs.append((clef.find("sign").text, clef.find("line").text,
                      change.text if change is not None else None))
    assert clefs == [("G", "2", None), ("G", "2", None), ("G", "2", "-1"), ("F", "4", None)]
    # music21 reads it as four parts, and the soprano keeps the corpus rhythm.
    score = music21.converter.parse(str(path))
    assert len(score.parts) == 4
    tune = load_tune(cid)
    written = [
        (float(n.offset), float(n.duration.quarterLength), n.pitch.midi if n.isNote else None)
        for n in score.parts[0].flatten().notesAndRests
    ]
    original = [(float(e.offset), float(e.length), e.midi) for e in tune.soprano]
    assert written[: len(original)] == original
    # A, T, B: one note per beat, the cadence held for the fermata's length.
    alto = [n for n in score.parts[1].stripTies().flatten().notes]
    assert [(float(n.offset), float(n.duration.quarterLength)) for n in alto] == [
        (float(b.offset), float(b.length)) for b in tune.beats
    ]


@pytest.mark.slow
def test_round_trip_and_checker_over_many_chorales(tmp_path):
    rng = random.Random(2026)
    chorales = usable_chorales()[::9]
    assert len(chorales) >= 20
    total = 0
    path = tmp_path / "score.musicxml"
    for cid in chorales:
        cap = capacity_report(cid)["capacity"]
        for _ in range(10):
            n = rng.randrange(cap)
            chords = encode(cid, n)
            assert decode(cid, chords) == n
            write_musicxml(cid, chords, path)
            back = read_musicxml(cid, path)
            assert back == chords
            assert decode(cid, back) == n
            assert check_musicxml(cid, path) == [], (cid, n)
            total += 1
    assert total >= 200


def test_checker_catches_a_broken_score(tmp_path):
    cid = 6
    chords = encode(cid, 0)
    path = tmp_path / "s.musicxml"
    write_musicxml(cid, chords, path)
    assert check_musicxml(cid, path) == []
    text = path.read_text()
    # Push the first bass note up a 5th: no longer the written chord.
    part = text.index('<part id="P4">')
    at = text.index("<octave>", part)
    broken = text[:at] + "<octave>4</octave>" + text[text.index("</octave>", at) + 9:]
    path.write_text(broken)
    assert check_musicxml(cid, path) != []


def test_checker_does_not_import_the_engine_rules():
    source = (ROOT / "chorale_engine" / "checker.py").read_text()
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]
    for banned in ("rules", "chords", "options"):
        assert not any(banned in line for line in imports), imports


@pytest.mark.slow
def test_cli_encode_is_byte_identical_across_processes(tmp_path):
    n = str(random.Random(2026).randrange(capacity_report(215)["capacity"]))  # a big integer
    a, b = tmp_path / "a.musicxml", tmp_path / "b.musicxml"
    for out in (a, b):
        result = run("encode", "--chorale", "215", "--n", n, "--out", str(out))
        assert result.returncode == 0, result.stderr
    assert a.read_bytes() == b.read_bytes()
    result = run("decode", "--chorale", "215", "--in", str(a))
    assert result.returncode == 0 and result.stdout.strip() == n
    assert check_musicxml(215, a) == []


@pytest.mark.slow
def test_cadences_and_opening_are_identical_across_processes():
    code = (
        "from chorale_engine.options import get_plan\n"
        "for cid in (6, 16, 58, 175, 215):\n"
        "    plan = get_plan(cid)\n"
        "    print(cid, plan.opening, plan.cadences)\n"
    )
    outs = [
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        for _ in range(2)
    ]
    assert outs[0].returncode == 0, outs[0].stderr
    assert outs[0].stdout == outs[1].stdout and outs[0].stdout.count("\n") == 5


@pytest.mark.slow
def test_cli_capacity_list_and_errors(tmp_path):
    result = run("capacity", "--chorale", "6")
    report = capacity_report(6)
    assert result.returncode == 0
    lines = result.stdout.strip().splitlines()
    assert len(lines) == len(report["phrases"]) + 1
    assert lines[0].startswith(f"phrase 0: {report['phrases'][0]['bits']} bits")
    assert lines[-1].startswith(f"total: {report['bits']} bits")
    result = run("list")
    assert result.returncode == 0
    assert result.stdout.splitlines()[0] == f"music21 {music21.__version__}"
    assert f"usable ({len(usable_chorales())}):" in result.stdout
    assert "  1: meter not 4/4" in result.stdout
    result = run("encode", "--chorale", "6", "--n", str(report["capacity"]), "--out", str(tmp_path / "x"))
    assert result.returncode == 1 and str(report["capacity"]) in result.stderr


def test_capacity_errors_per_phrase_and_whole_piece():
    for cid in (6, 58):
        cap = capacity_report(cid)["capacity"]
        for bad in (cap, -1):
            with pytest.raises(CapacityError):
                encode(cid, bad)
        for p in range(load_tune(cid).n_phrases):
            for bad in (phrase_capacity(cid, p), -1):
                with pytest.raises(CapacityError):
                    encode_phrase(cid, p, bad)


def test_altered_chord_in_musicxml_raises_with_indices(tmp_path):
    cid = 16
    tune = load_tune(cid)
    chords = encode(cid, 123456)
    path = tmp_path / "s.musicxml"
    write_musicxml(cid, chords, path)
    back = read_musicxml(cid, path)
    beat = tune.beat(2, 3)
    wrong = list(back)
    c = wrong[beat.index]
    wrong[beat.index] = (c[0], c[1], c[2], c[3] + 1)
    with pytest.raises(ChordNotInOptions) as err:
        decode(cid, wrong)
    assert (err.value.phrase_index, err.value.beat_index) == (2, 3)
    start, end = tune.phrases[2]
    with pytest.raises(ChordNotInOptions) as err:
        decode_phrase(cid, 2, wrong[start:end])
    assert (err.value.phrase_index, err.value.beat_index) == (2, 3)
