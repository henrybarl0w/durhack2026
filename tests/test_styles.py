"""The style switch: original encoding and the fancier writing, decoded without being told which."""

import random
import subprocess
import sys
from pathlib import Path

import pytest

import chorale_engine as ce
from chorale_engine.weights import default_weights

ROOT = Path(__file__).resolve().parent.parent


def test_original_style_is_exactly_the_original_encoding(tmp_path):
    n = 123456789
    assert ce.style_weights("original") is default_weights
    assert ce.encode_style(16, n, "original") == ce.encode(16, n)
    plain, styled = tmp_path / "a.musicxml", tmp_path / "b.musicxml"
    ce.write_musicxml(16, ce.encode(16, n), plain)
    ce.write_musicxml(16, ce.encode_style(16, n, "original"), styled, style="original")
    assert plain.read_bytes() == styled.read_bytes()
    with pytest.raises(ValueError):
        ce.style_weights("baroque")


@pytest.mark.parametrize("cid", [6, 16, 58])
def test_both_styles_decode_and_name_their_style(cid, tmp_path):
    rng = random.Random(cid)
    scores = {}
    for style in ce.STYLES:
        n = rng.randrange(ce.style_capacity(cid, style)["capacity"])
        chords = ce.encode_style(cid, n, style)
        path = tmp_path / f"{style}.musicxml"
        ce.write_musicxml(cid, chords, path, style=style)
        assert ce.decode_musicxml_with_style(path) == (cid, n, style)
        assert ce.decode_musicxml(path) == (cid, n)
        assert ce.decode_musicxml(path, ce.style_weights(style)) == (cid, n)
        scores[style] = path.read_bytes()
    assert scores["original"] != scores["fancy"]


def test_fancy_ends_on_a_perfect_cadence_where_the_tune_allows():
    from chorale_engine.chords import analyse, beat_context
    from chorale_engine.options import get_plan

    weights = ce.style_weights("fancy")
    for cid in (6, 16, 47, 58):
        if not get_plan(cid, weights).style.final_dominant:
            continue
        tune = ce.load_tune(cid)
        chords = ce.encode_style(cid, 7, "fancy")
        info = analyse(beat_context(tune, len(chords) - 2), chords[-2])
        assert info.ctype.dominant, cid


def test_cli_style_switch(tmp_path):
    for style in ce.STYLES:
        out = tmp_path / f"{style}.musicxml"
        result = subprocess.run(
            [sys.executable, "-m", "chorale_engine", "encode", "--chorale", "16", "--n", "4242",
             "--style", style, "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([sys.executable, "-m", "chorale_engine", "decode", "--in", str(out)],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() == "4242"
        assert f"{style} style" in result.stderr
