"""Milestone 5 and acceptance: MusicXML, checker, CLI, determinism."""

import random
import subprocess
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction
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
    decode_musicxml,
    identify_chorale,
    matching_chorales,
    usable_chorales,
    write_musicxml,
    DecodeError,
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
    assert root.find("identification") is None  # no encoding date
    parts = root.findall("part")
    assert len(parts) == 1 and parts[0].get("id") == "P1"
    attrs = parts[0].find("measure/attributes")
    assert attrs.find("staves").text == "2"
    assert [(c.get("number"), c.find("sign").text, c.find("line").text) for c in attrs.findall("clef")] == [
        ("1", "G", "2"), ("2", "F", "4"),
    ]
    # Voice 1-2 on the treble staff with stems up/down, voice 3-4 on the bass staff.
    layout = {
        (n.find("voice").text, n.find("staff").text, n.find("stem").text)
        for n in parts[0].iter("note") if n.find("pitch") is not None
    }
    assert layout == {("1", "1", "up"), ("2", "1", "down"), ("3", "2", "up"), ("4", "2", "down")}
    # music21 reads it, the soprano (voice 1) keeps the corpus rhythm.
    score = music21.converter.parse(str(path))
    tune = load_tune(cid)
    treble = score.parts[0]
    soprano = [v for v in treble.recurse().getElementsByClass("Voice") if v.id == "1"]
    written = []
    for v in soprano:
        for n in v.notesAndRests:
            written.append((float(n.getOffsetInHierarchy(treble)), float(n.duration.quarterLength),
                            n.pitch.midi if n.isNote else None))
    written.sort(key=lambda r: r[0])
    original = [(float(e.offset), float(e.length), e.midi) for e in tune.soprano]
    assert written[: len(original)] == original


def test_shared_notes_read_back_from_one_notehead(tmp_path):
    """Where S = A or T = B, a score with only one notehead still decodes."""
    found = 0
    for cid in (6, 16, 58, 215):
        tune = load_tune(cid)
        n = capacity_report(cid)["capacity"] // 3
        chords = encode(cid, n)
        shared = [i for i, c in enumerate(chords) if c[0] == c[1] or c[2] == c[3]]
        if not shared:
            continue
        found += len(shared)
        path = tmp_path / f"{cid}.musicxml"
        write_musicxml(cid, chords, path)
        text = path.read_text()
        path2 = tmp_path / f"{cid}-single.musicxml"
        path2.write_text(_drop_shared(text, tune, chords))
        assert read_musicxml(cid, path2) == chords
        assert decode_musicxml(path2) == (cid, n)
    assert found > 0


def _drop_shared(text, tune, chords):
    """Turn shared alto/bass notes at beats into rests (one notehead left on the staff)."""
    root = ET.fromstring(text)
    part = root.find("part")
    divisions = int(part.find("measure/attributes/divisions").text)
    drop = {}
    for beat, c in zip(tune.beats, chords):
        if c[0] == c[1]:
            drop[("2", beat.offset)] = True
        if c[2] == c[3]:
            drop[("4", beat.offset)] = True
    for measure, info in zip(part.findall("measure"), tune.measures):
        at = Fraction(0)
        for el in measure:
            if el.tag == "backup":
                at -= Fraction(int(el.find("duration").text), divisions)
            elif el.tag == "note":
                length = Fraction(int(el.find("duration").text), divisions)
                onset = info.offset + at
                if el.find("pitch") is not None and drop.get((el.find("voice").text, onset)):
                    el.remove(el.find("pitch"))
                    el.insert(0, ET.Element("rest"))
                    for tag in ("tie", "stem", "notations"):
                        for sub in el.findall(tag):
                            el.remove(sub)
                at += length
    return ET.tostring(root, encoding="unicode")


def test_decode_recognises_the_chorale(tmp_path):
    rng = random.Random(7)
    for cid in (6, 16, 47, 58, 215, 227):
        n = rng.randrange(capacity_report(cid)["capacity"])
        path = tmp_path / f"{cid}.musicxml"
        write_musicxml(cid, encode(cid, n), path)
        assert cid in matching_chorales(path)
        assert decode_musicxml(path) == (cid, n)
        assert identify_chorale(path) == cid


def test_duplicate_tunes_are_told_apart_by_decoding(tmp_path):
    # 80, 98, 286 and 367 share one soprano line, pitch for pitch.
    group = [cid for cid in (80, 98, 286, 367) if cid in usable_chorales()]
    assert len(group) >= 2
    rng = random.Random(8)
    for cid in group:
        n = rng.randrange(capacity_report(cid)["capacity"])
        path = tmp_path / f"{cid}.musicxml"
        write_musicxml(cid, encode(cid, n), path)
        assert set(group) <= set(matching_chorales(path))
        found, value = decode_musicxml(path)
        assert value == n and found in group


def test_unknown_tune_is_a_decode_error(tmp_path):
    path = tmp_path / "s.musicxml"
    write_musicxml(6, encode(6, 0), path)
    text = path.read_text().replace("<step>F</step>", "<step>E</step>", 1)  # change the first soprano note
    path.write_text(text)
    with pytest.raises(DecodeError):
        decode_musicxml(path)


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
    # Move the first bass note (voice 4) up two octaves: no longer the written chord.
    part = text.index("<voice>4</voice>")
    at = text.rindex("<octave>", 0, part)
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
