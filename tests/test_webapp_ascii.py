import io
import re
import wave
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from webapp import app as webapp


@pytest.mark.parametrize("text", ["Hello, world!", " ~", "trailing spaces  "])
def test_ascii_chunk_round_trip(text):
    assert webapp.unpack_ascii_chunk(webapp.pack_ascii_chunk(text)) == text


@pytest.mark.parametrize("text", ["", "line\nbreak", "café", "\x7f"])
def test_ascii_message_rejects_non_printable_or_non_ascii(text):
    with pytest.raises(ValueError):
        webapp.validate_ascii_message(text)


def test_ascii_encoding_splits_at_maximum_chorale_capacity(monkeypatch):
    monkeypatch.setattr(webapp, "CHECKSUM", False)  # this test describes the layout without a checksum
    capacity = 1 << 60
    monkeypatch.setattr(webapp, "chorale_pool", lambda: {"original": {10: capacity, 11: capacity // 2}})
    monkeypatch.setattr(webapp, "pick_chorale", lambda n, style: 10 if n < capacity else 11)
    monkeypatch.setattr(webapp, "engine_encode", lambda cid, n, style: n)

    chunk_length = webapp.ascii_chunk_length("original")
    assert webapp.pack_ascii_chunk("~" * chunk_length) < capacity
    assert webapp.pack_ascii_chunk("~" * (chunk_length + 1)) >= capacity

    message = "A B C"
    parts = webapp.encode_ascii(message)
    recovered = "".join(webapp.unpack_ascii_chunk(chords) for _cid, chords in parts)
    assert len(parts) == 2
    assert recovered == message


def test_ascii_zip_decodes_parts_in_numeric_order(monkeypatch):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("part-0002.musicxml", str(webapp.pack_ascii_chunk("second")))
        bundle.writestr("part-0001.musicxml", str(webapp.pack_ascii_chunk("first ")))

    monkeypatch.setattr(
        webapp, "decode_musicxml_with_style",
        lambda path: (10, int(Path(path).read_text()), "original"),
    )
    message, chorales = webapp.decode_ascii_archive(archive.getvalue())

    assert message == "first second"
    assert chorales == [10, 10]


def test_combine_pngs_stacks_every_part_vertically():
    parts = []
    for size, color in [((2, 3), "red"), ((4, 2), "blue")]:
        out = io.BytesIO()
        Image.new("RGB", size, color).save(out, format="PNG")
        parts.append(out.getvalue())

    combined = Image.open(io.BytesIO(webapp.combine_pngs(parts)))

    assert combined.size == (4, 5)
    assert combined.getpixel((1, 1)) == (255, 0, 0)
    assert combined.getpixel((1, 4)) == (0, 0, 255)


def test_merge_wav_files_concatenates_frames():
    def make_wav(frames):
        out = io.BytesIO()
        with wave.open(out, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(16000)
            wav.writeframes(frames)
        return out.getvalue()

    combined = webapp.merge_wav_files([make_wav(b"\x01\x00"), make_wav(b"\x02\x00\x03\x00")])

    with wave.open(io.BytesIO(combined), "rb") as wav:
        assert wav.getnframes() == 3
        assert wav.readframes(3) == b"\x01\x00\x02\x00\x03\x00"


def test_ascii_payload_reports_part_count_and_total_bits(monkeypatch):
    monkeypatch.setattr(webapp, "capacity_bits", lambda style: 32)
    monkeypatch.setattr(webapp, "chorale_pool", lambda: {"original": {10: 1 << 60}})
    monkeypatch.setattr(webapp, "ascii_chunk_length", lambda style: 4)

    response = webapp.app.test_client().get(
        "/payload", query_string={"message": "ABCDEFG", "method": "ascii"}
    )

    assert response.json["chorales"] == 2
    assert response.json["bits"] == 7 * 7 + 2 * webapp.ASCII_MAGIC_BITS


def test_encode_page_template_renders():
    response = webapp.app.test_client().get("/")

    assert response.status_code == 200
    assert b"Printable ASCII" in response.data
    assert b"Default" in response.data
    assert b"Error correction (checksum)" in response.data


def test_ascii_encode_route_returns_ordered_musicxml_zip(monkeypatch):
    monkeypatch.setattr(webapp, "encode_ascii", lambda text, style: [(10, ["a"]), (11, ["b"])])
    monkeypatch.setattr(webapp, "musicxml_string", lambda cid, chords, **kwargs: f"score-{cid}")
    monkeypatch.setattr(webapp, "style_writing", lambda style: {})
    monkeypatch.setattr(webapp.renderer, "writeScore", lambda xml, *args: f"{xml}-png".encode())
    monkeypatch.setattr(webapp, "chords_to_wav", lambda chords, cid, style: f"wav-{cid}".encode())
    monkeypatch.setattr(webapp, "combine_pngs", lambda images: b"combined-png")
    monkeypatch.setattr(webapp, "merge_wav_files", lambda files: b"combined-wav")
    client = webapp.app.test_client()

    response = client.post(
        "/encode",
        data={"secret_message": "a longer message", "encoding_method": "ascii"},
    )
    match = re.search(rb'href="/download/([^"]+\.zip)" download>Download MusicXML ZIP', response.data)

    assert response.status_code == 200
    assert match
    assert b"Download PNG" in response.data
    assert b"Download WAV" in response.data
    assert b"All parts" in response.data  # the part tabs end with the combined page
    assert b"Download all PNGs" not in response.data
    assert b"Download all WAVs" not in response.data
    assert b"<img" in response.data and b"<audio controls" in response.data
    downloaded = client.get("/download/" + match.group(1).decode())
    with zipfile.ZipFile(io.BytesIO(downloaded.data)) as bundle:
        assert bundle.namelist() == ["part-0001.musicxml", "part-0002.musicxml"]
        assert bundle.read("part-0001.musicxml") == b"score-10"
        assert bundle.read("part-0002.musicxml") == b"score-11"
    links = {
        label.decode(): href.decode()
        for href, label in re.findall(rb'<a class="dl" href="/download/([^"]+)" download aria-label="([^"]+)"', response.data)
    }
    assert client.get("/download/" + links["Download PNG"]).data == b"combined-png"
    assert client.get("/download/" + links["Download WAV"]).data == b"combined-wav"
