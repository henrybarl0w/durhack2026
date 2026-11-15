"""Checksum flag: 16-bit CRC on the message, and repair of misread chords."""

import random
import subprocess
import sys
from pathlib import Path

import pytest

import chorale_engine as ce
from chorale_engine.checksum import CHECK_BITS, crc16, decode_checked, split_check, with_check
from chorale_engine.options import get_plan

ROOT = Path(__file__).resolve().parent.parent
CID = 215  # the largest chorale: room for the checksum and a three-word message


def test_crc_attach_and_split():
    assert crc16(0) != crc16(1)
    for n in (0, 1, 255, 256, 2 ** 45 + 7):
        value = with_check(n)
        assert value >> CHECK_BITS == n and split_check(value) == n
        assert split_check(value ^ 1) is None  # one flipped bit is caught
    assert crc16(0x313233) == 0x5BCE  # CRC-16/CCITT-FALSE check value for "123" bytes


def test_capacity_is_reduced_by_the_check_bits():
    plain = ce.style_capacity(CID, "original")
    checked = ce.style_capacity(CID, "original", checksum=True)
    assert checked["capacity"] == plain["capacity"] >> CHECK_BITS
    assert checked["check_bits"] == CHECK_BITS
    with pytest.raises(ce.CapacityError):
        ce.encode_style(CID, checked["capacity"], "original", checksum=True)


def _damage(chords, beats, rng):
    """Replace the chord at each beat by a wrong one (another option, or a bass a step off)."""
    out = list(chords)
    for b in beats:
        s, a, t, bass = out[b]
        out[b] = (s, a, t, bass + rng.choice([-1, 1]))
    return out


@pytest.mark.parametrize("style", ce.STYLES)
def test_repairs_one_and_two_misread_chords(style):
    rng = random.Random(4)
    tune = ce.load_tune(CID)
    cap = ce.style_capacity(CID, style, checksum=True)["capacity"]
    for trial in range(10):
        n = rng.randrange(cap)
        chords = ce.encode_style(CID, n, style, checksum=True)
        weights = ce.style_weights(style)
        assert decode_checked(CID, chords, weights).n == n
        # one error anywhere, and two errors (same phrase or not)
        for k in (1, 2):
            spots = rng.sample(range(len(chords) - 1), k)
            result = decode_checked(CID, _damage(chords, spots, rng), weights)
            assert result.n == n
            assert result.chords == chords
            fixed_beats = {tune.phrases[p][0] + b for p, b in result.fixed}
            assert set(spots) <= fixed_beats


def test_unreadable_beats_are_filled_in():
    n = 123456789
    chords = ce.encode_style(CID, n, "original", checksum=True)
    holes = list(chords)
    holes[3] = None
    holes[40] = None
    assert decode_checked(CID, holes).n == n


def test_too_many_errors_are_refused_not_guessed():
    rng = random.Random(9)
    n = 987654321
    chords = ce.encode_style(CID, n, "original", checksum=True)
    tune = ce.load_tune(CID)
    refused = wrong = 0
    for trial in range(15):
        # five errors in five different phrases, away from single-option cadence beats
        spots = []
        for p in rng.sample(range(tune.n_phrases), 5):
            s, e = tune.phrases[p]
            spots.append(rng.randrange(s + 1, e - 1))
        try:
            wrong += decode_checked(CID, _damage(chords, spots, rng)).n != n
        except ce.DecodeError:
            refused += 1
    assert wrong == 0 and refused > 0


def test_plain_score_is_not_mistaken_for_a_checked_one():
    # without the checksum, the number almost never carries a valid check by chance
    n = 31415926535
    chords = ce.encode(CID, n)
    with pytest.raises(ce.DecodeError):
        decode_checked(CID, chords, max_errors=0)


def test_musicxml_and_cli_round_trip(tmp_path):
    n = 4242
    for style in ce.STYLES:
        path = tmp_path / f"{style}.musicxml"
        ce.write_musicxml(CID, ce.encode_style(CID, n, style, checksum=True), path, style=style)
        reading = ce.decode_score(path, checksum=True)
        assert (reading.chorale_id, reading.n, reading.style, reading.repaired) == (CID, n, style, [])
        assert ce.decode_musicxml(path, checksum=True) == (CID, n)
        out = tmp_path / f"cli_{style}.musicxml"
        result = subprocess.run([sys.executable, "-m", "chorale_engine", "encode", "--chorale", str(CID), "--n", str(n),
                                 "--style", style, "--checksum", "--out", str(out)], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        result = subprocess.run([sys.executable, "-m", "chorale_engine", "decode", "--in", str(out), "--checksum"],
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0 and result.stdout.strip() == str(n), result.stderr
    result = subprocess.run([sys.executable, "-m", "chorale_engine", "capacity", "--chorale", str(CID), "--checksum"],
                            cwd=ROOT, capture_output=True, text=True)
    assert "with checksum:" in result.stdout
