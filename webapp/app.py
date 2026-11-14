"""Web front end: up to 3 words from the word list -> chorale 215 (with error
correction) -> score / audio, and back from the MusicXML file.

Everything is held in memory. Run from the project root or from webapp/.
"""
import io
import math
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
from chorale_engine import CapacityError, ChoraleRejected, DecodeError, load_tune, read_musicxml
from chorale_engine.ecc import best_parity, decode_words, encode_words
from chorale_engine.musicxml_io import musicxml_string

CHORALE_ID = 215
MAX_WORDS = 3
MAX_PARITY = 2  # damaged phrases we try to be able to repair (lowered automatically if it does not fit)
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


@lru_cache(maxsize=1)
def parity():
    """Repairable phrases for chorale 215: the most that still fits MAX_WORDS words."""
    needed = (((len(WORDS) + 1) ** MAX_WORDS - 1).bit_length() + 7) // 8
    r = best_parity(CHORALE_ID, needed, MAX_PARITY)
    if r is None:
        raise CapacityError(f"Chorale {CHORALE_ID} cannot hold {MAX_WORDS} words from this word list.")
    print(f"Chorale {CHORALE_ID}: can repair up to {r} damaged phrase(s).")
    return r


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


# ------------------------------------------------------------ chords -> audio
def chords_to_wav(chords, bpm=76, rate=16000):
    """A small additive-synth rendering of the chords, as WAV bytes."""
    tune = load_tune(CHORALE_ID)
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


# -------------------------------------------------------------------- routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/encode", methods=["POST"])
def encode():
    try:
        indices = words_to_indices(request.form.get("secret_message", ""))
        chords = encode_words(CHORALE_ID, indices, len(WORDS), parity())
        xml = musicxml_string(CHORALE_ID, chords)
        png = renderer.writeScore(xml, None, "png", 2000, 1.0)
        files = {
            "png_file": stash(png, "png", "image/png"),
            "xml_file": stash(xml, "musicxml", "application/vnd.recordare.musicxml+xml"),
            "wav_file": stash(chords_to_wav(chords), "wav", "audio/wav"),
        }
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="encode", error=str(exc))
    return render_template("index.html", active_tab="encode", **files)


@app.route("/decode", methods=["POST"])
def decode():
    upload = request.files.get("encoded_file")
    if not upload or not upload.filename:
        return render_template("index.html", active_tab="decode", error="No file provided.")
    try:
        if Path(upload.filename).suffix.lower() not in (".musicxml", ".xml"):
            raise ValueError("Upload the .musicxml file made by this tool "
                             "(the PNG and WAV can't be decoded).")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.musicxml"
            path.write_bytes(upload.read())
            chords = read_musicxml(CHORALE_ID, str(path))
        indices, repaired, bad = decode_words(CHORALE_ID, chords, len(WORDS), parity())
        if not 1 <= len(indices) <= MAX_WORDS:
            raise DecodeError("No message from this tool was found in that file.")
        text = " ".join(WORDS[i] for i in indices)
        wav = stash(chords_to_wav(repaired), "wav", "audio/wav")
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="decode", error=str(exc))
    note = (f"Repaired damaged phrase(s) {', '.join(str(p + 1) for p in bad)}. The message is intact."
            if bad else "No damage found.")
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