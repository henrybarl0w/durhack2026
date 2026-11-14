"""Web front end for the chorale steganography tool.

Encode:  text -> integer -> error-corrected chords -> MusicXML / PNG / MIDI / WAV
Decode:  MIDI or MusicXML -> chords -> error-corrected integer -> text

Everything is held in memory (no files are written for downloads).
"""
import io
import math
import os
import secrets
import sys
import tempfile
import threading
import wave
import zlib
from array import array
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, abort, render_template, request, send_file

import renderer
from chorale_engine import (
    CapacityError,
    ChoraleRejected,
    DecodeError,
    identify_chorale,
    load_tune,
    read_musicxml,
    usable_chorales,
)
from chorale_engine.ecc import decode_ecc_report, ecc_capacity, encode_ecc
from chorale_engine.musicxml_io import musicxml_string

# How many damaged phrases the decoder can repair. Must be the same for
# encoding and decoding. Lower it (1) if messages do not fit any chorale.
PARITY = int(os.environ.get("CHORALE_PARITY", "2"))
MAX_WORDS = 3
TICKS_PER_BEAT = 480
USER_ERRORS = (CapacityError, ChoraleRejected, DecodeError, renderer.RendererError, ValueError)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


# ----------------------------------------------------------- in-memory files
_FILES: "OrderedDict[str, tuple[bytes, str]]" = OrderedDict()
_LOCK = threading.Lock()
_MAX_FILES = 64


def stash(data, ext, mimetype):
    """Keep a generated file in RAM and return the name used in /download/<name>."""
    name = f"{secrets.token_urlsafe(9)}.{ext}"
    with _LOCK:
        _FILES[name] = (data if isinstance(data, bytes) else data.encode("utf-8"), mimetype)
        while len(_FILES) > _MAX_FILES:
            _FILES.popitem(last=False)
    return name


# -------------------------------------------------------- message <-> integer
def clean_message(raw):
    text = " ".join(raw.replace("\x00", "").split())
    if not text:
        raise ValueError("Type a message first.")
    if len(text.split()) > MAX_WORDS:
        raise ValueError(f"Messages can be up to {MAX_WORDS} words.")
    return text


def pack(text):
    """[length byte][utf-8 bytes], read as a little-endian integer."""
    raw = text.encode("utf-8")
    if len(raw) > 255:
        raise ValueError("Message is too long.")
    return int.from_bytes(bytes([len(raw)]) + raw, "little")


def unpack(n):
    data = n.to_bytes((n.bit_length() + 7) // 8, "little")
    try:
        if not data or data[0] != len(data) - 1:
            raise ValueError
        return data[1:].decode("utf-8")
    except ValueError:  # includes UnicodeDecodeError
        raise DecodeError("No message from this tool was found in that file.") from None


# ------------------------------------------------------------ chorale choice
@lru_cache(maxsize=1)
def _chorale_ids():
    return sorted(usable_chorales())


@lru_cache(maxsize=None)
def _capacity_bytes(cid):
    return ecc_capacity(cid, PARITY)["bytes"]


def pick_chorale(needed_bytes, seed):
    """A chorale (varied by message) with room for the message plus parity."""
    ids = _chorale_ids()
    if not ids:
        raise ChoraleRejected(None, "no usable chorales")
    best = 0
    for i in range(len(ids)):
        cid = ids[(seed + i) % len(ids)]
        room = _capacity_bytes(cid)
        if room >= needed_bytes:
            return cid
        best = max(best, room)
    raise CapacityError(
        f"That message needs {needed_bytes} bytes but no chorale holds more than {best} "
        f"with {PARITY} repairable phrases. Use a shorter message or set CHORALE_PARITY lower."
    )


# --------------------------------------------------------- chords -> outputs
def chords_to_midi(tune, chords):
    notes = []
    for beat, chord in zip(tune.beats, chords):
        start = int(beat.offset * TICKS_PER_BEAT)
        end = int((beat.offset + beat.length) * TICKS_PER_BEAT)
        for midi in sorted(set(chord)):
            notes.append(renderer.Note(midi, start, end))
    return renderer.writeMidi(notes, TICKS_PER_BEAT)


def chords_to_wav(tune, chords, bpm=76, rate=16000):
    """A small additive-synth rendering of the chords, as WAV bytes."""
    per_quarter = 60.0 / bpm * rate
    cache = {}

    def tone(midi, n):
        if (midi, n) not in cache:
            w = 2 * math.pi * 440.0 * 2 ** ((midi - 69) / 12) / rate
            atk, rel = min(int(0.01 * rate), n // 4), min(int(0.08 * rate), n // 3)
            buf = []
            for i in range(n):
                env = i / atk if i < atk else (n - i) / rel if i > n - rel else 1.0
                x = i * w
                buf.append(env * (math.sin(x) + 0.4 * math.sin(2 * x) + 0.15 * math.sin(3 * x)))
            cache[(midi, n)] = buf
        return cache[(midi, n)]

    pcm = array("h")
    for beat, chord in zip(tune.beats, chords):
        start = round(beat.offset * per_quarter)
        end = round((beat.offset + beat.length) * per_quarter)
        if start > len(pcm):
            pcm.extend([0] * (start - len(pcm)))
        voices = [tone(m, end - start) for m in chord]
        pcm.extend(int(3800 * sum(s)) for s in zip(*voices))
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return out.getvalue()


def build_downloads(cid, chords):
    xml = musicxml_string(cid, chords)
    tune = load_tune(cid)
    return {
        "xml_file": stash(xml, "musicxml", "application/vnd.recordare.musicxml+xml"),
        "png_file": stash(renderer.writeScore(xml, None, "png", 2000, 1.0), "png", "image/png"),
        "midi_file": stash(chords_to_midi(tune, chords), "mid", "audio/midi"),
        "wav_file": stash(chords_to_wav(tune, chords), "wav", "audio/wav"),
    }


# ----------------------------------------------------------- decode pipeline
def _identify(path):
    result = identify_chorale(path)
    return int(getattr(result, "chorale_id", result))


def decode_upload(filename, data):
    ext = Path(filename).suffix.lower()
    if ext not in (".mid", ".midi", ".musicxml", ".xml"):
        raise ValueError("Upload the .mid or .musicxml file made by this tool "
                         "(images and WAV files can't be decoded).")
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "score.musicxml"
        if ext in (".mid", ".midi"):
            mid_path = Path(tmp) / "score.mid"
            mid_path.write_bytes(data)
            xml_path.write_text(renderer.midiToMusicXml(str(mid_path)), encoding="utf-8")
        else:
            xml_path.write_bytes(data)
        cid = _identify(str(xml_path))
        chords = read_musicxml(cid, str(xml_path))
    n, repaired, bad = decode_ecc_report(cid, chords, PARITY)
    return unpack(n), cid, repaired, bad


# -------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/encode", methods=["POST"])
def encode():
    try:
        text = clean_message(request.form.get("secret_message", ""))
        n = pack(text)
        cid = pick_chorale((n.bit_length() + 7) // 8, zlib.crc32(text.encode("utf-8")))
        files = build_downloads(cid, encode_ecc(cid, n, PARITY))
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="encode", error=str(exc))
    return render_template("index.html", active_tab="encode", **files)


@app.route("/decode", methods=["POST"])
def decode():
    upload = request.files.get("encoded_file")
    if not upload or not upload.filename:
        return render_template("index.html", active_tab="decode", error="No file provided.")
    try:
        text, cid, repaired, bad = decode_upload(upload.filename, upload.read())
        wav = stash(chords_to_wav(load_tune(cid), repaired), "wav", "audio/wav")
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="decode", error=str(exc))
    if bad:
        shown = ", ".join(str(p + 1) for p in bad)
        note = f"Chorale {cid}: repaired damaged phrase(s) {shown}. The message is intact."
    else:
        note = f"Chorale {cid}: no damage found."
    return render_template("index.html", active_tab="decode", decoded_result=text,
                           repair_note=note, wav_file=wav)


@app.route("/download/<name>")
def download(name):
    with _LOCK:
        item = _FILES.get(name)
    if item is None:
        abort(404)
    data, mimetype = item
    return send_file(io.BytesIO(data), mimetype=mimetype, download_name=name)


if __name__ == "__main__":
    app.run(debug=True)