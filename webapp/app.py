"""Web front end: up to 3 words from the word list -> chorale 215 -> MusicXML
(plus PNG and WAV extras), and back from the MusicXML file.

Two encodings, chosen automatically at start-up:
  * error-corrected: used when chorale 215 has room for it (repairs damaged phrases);
  * plain: exactly what test.py does (index + 10000, five digits per word).
The decoder reads both. Set CHORALE_ECC=0 to force plain.
"""
import io
import math
import os
import secrets
import sys
import tempfile
import threading
import wave
from array import array
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, abort, render_template, request, send_file

import renderer
from chorale_engine import (
    CapacityError, ChoraleRejected, DecodeError, capacity_report, decode_musicxml,
    encode as engine_encode, load_tune, read_musicxml,
)
from chorale_engine.ecc import decode_words, ecc_capacity, encode_words
from chorale_engine.musicxml_io import musicxml_string

CHORALE_ID = 215
MAX_WORDS = 3
MAX_PARITY = 2
USER_ERRORS = (CapacityError, ChoraleRejected, DecodeError, renderer.RendererError, ValueError)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024


# ---------------------------------------------------------------- word list
def _load_words():
    for folder in (ROOT, Path(__file__).resolve().parent):
        path = folder / "durhack_word_list.csv"
        if path.exists():
            return [w.strip() for w in path.read_text(encoding="utf-8-sig").splitlines()]
    raise FileNotFoundError("durhack_word_list.csv not found next to test.py")


WORDS = _load_words()
LOOKUP = {}
for _i, _w in enumerate(WORDS):
    if _w:
        LOOKUP.setdefault(_w.lower(), _i)


def words_to_indices(raw):
    words = raw.split()
    if not words:
        raise ValueError("Type a message first.")
    if len(words) > MAX_WORDS:
        raise ValueError(f"Messages can be up to {MAX_WORDS} words.")
    out = []
    for w in words:
        if w.lower() not in LOOKUP:
            raise ValueError(f"'{w}' is not in the word list.")
        out.append(LOOKUP[w.lower()])
    return out


# ------------------------------------------------------------- encoding mode
@lru_cache(maxsize=1)
def mode():
    """('ecc', repairable phrases) if it fits chorale 215, otherwise ('plain', 0)."""
    if os.environ.get("CHORALE_ECC") != "0":
        needed = (((len(WORDS) + 1) ** MAX_WORDS - 1).bit_length() + 7) // 8
        for r in range(MAX_PARITY, 0, -1):
            try:
                if ecc_capacity(CHORALE_ID, r)["bytes"] >= needed:
                    print(f"Chorale {CHORALE_ID}: error-corrected mode, repairs up to {r} damaged phrase(s).")
                    return ("ecc", r)
            except Exception as exc:  # fall back rather than fail
                print("Error-correction capacity check failed:", exc)
                break
    try:
        bits = capacity_report(CHORALE_ID)["bits"]
        print(f"Chorale {CHORALE_ID}: plain mode (it holds {bits} bits, too little for error correction).")
    except Exception:
        print(f"Chorale {CHORALE_ID}: plain mode.")
    return ("plain", 0)


def encode_indices(indices):
    kind, r = mode()
    if kind == "ecc":
        return encode_words(CHORALE_ID, indices, len(WORDS), r)
    if any(i + 10000 > 99999 for i in indices):
        raise ValueError("The word list is too long for plain mode.")
    return engine_encode(CHORALE_ID, int("".join(str(i + 10000) for i in indices)))


def decode_file(path):
    """(word indices, chords or None, repaired phrase indices, chorale id)."""
    kind, r = mode()
    ecc_error = None
    if kind == "ecc":
        try:
            idx, repaired, bad = decode_words(CHORALE_ID, read_musicxml(CHORALE_ID, path), len(WORDS), r)
            if 1 <= len(idx) <= MAX_WORDS:
                return idx, repaired, bad, CHORALE_ID
        except (DecodeError, ValueError) as exc:
            ecc_error = exc
    try:
        cid, n = decode_musicxml(path)
        digits = str(n)
        idx = [int(digits[i:i + 5]) - 10000 for i in range(0, len(digits), 5)]
        if len(digits) % 5 or not 1 <= len(idx) <= MAX_WORDS or any(not 0 <= i < len(WORDS) for i in idx):
            raise DecodeError("No message from this tool was found in that file.")
    except (DecodeError, ValueError):
        raise ecc_error or DecodeError("No message from this tool was found in that file.")
    cid = cid if isinstance(cid, int) else CHORALE_ID
    try:
        chords = read_musicxml(cid, path)
    except Exception:
        chords = None
    return idx, chords, [], cid


# ----------------------------------------------------------- in-memory files
_FILES = OrderedDict()
_LOCK = threading.Lock()


def stash(data, ext, mimetype):
    name = f"{secrets.token_urlsafe(9)}.{ext}"
    with _LOCK:
        _FILES[name] = (data if isinstance(data, bytes) else data.encode("utf-8"), mimetype)
        while len(_FILES) > 64:
            _FILES.popitem(last=False)
    return name


def chords_to_wav(chords, cid=CHORALE_ID, bpm=76, rate=16000):
    """A small additive-synth rendering of the chords, as WAV bytes."""
    tune = load_tune(cid)
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


def optional(label, make):
    """Run an extra (PNG / WAV); if it fails, log it and carry on without it."""
    try:
        return make()
    except Exception:
        app.logger.exception("%s could not be made", label)
        return None


# -------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/encode", methods=["POST"])
def encode():
    try:
        chords = encode_indices(words_to_indices(request.form.get("secret_message", "")))
        xml = musicxml_string(CHORALE_ID, chords)
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="encode", error=str(exc))
    png = optional("PNG", lambda: renderer.writeScore(xml, None, "png", 2000, 1.0))
    wav = optional("WAV", lambda: chords_to_wav(chords))
    return render_template(
        "index.html", active_tab="encode",
        xml_file=stash(xml, "musicxml", "application/vnd.recordare.musicxml+xml"),
        png_file=stash(png, "png", "image/png") if png else None,
        wav_file=stash(wav, "wav", "audio/wav") if wav else None,
    )


@app.route("/decode", methods=["POST"])
def decode():
    upload = request.files.get("encoded_file")
    if not upload or not upload.filename:
        return render_template("index.html", active_tab="decode", error="No file provided.")
    try:
        if Path(upload.filename).suffix.lower() not in (".musicxml", ".xml"):
            raise ValueError("Upload a .musicxml (or .xml) file.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.musicxml"
            path.write_bytes(upload.read())
            indices, chords, bad, cid = decode_file(str(path))
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="decode", error=str(exc))
    wav = optional("WAV", lambda: chords_to_wav(chords, cid)) if chords else None
    note = (f"Repaired damaged phrase(s) {', '.join(str(p + 1) for p in bad)}. The message is intact."
            if bad else "No damage found.")
    return render_template("index.html", active_tab="decode",
                           decoded_result=" ".join(WORDS[i] for i in indices), repair_note=note,
                           wav_file=stash(wav, "wav", "audio/wav") if wav else None)


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
