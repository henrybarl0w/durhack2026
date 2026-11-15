"""Web front end: up to 3 words from the word list -> a Bach chorale -> MusicXML
(plus PNG and WAV extras), and back from the MusicXML file.

In plain mode each message gets a chorale picked at random from those that can hold it
in the chosen style; the decoder recognises the chorale from its soprano line. In
error-corrected mode the chorale stays fixed (CHORALE_ID).

Two encodings, chosen automatically at start-up:
  * error-corrected: used when chorale 215 has room for it (repairs damaged phrases);
  * plain: exactly what test.py does (index + 10000, five digits per word).
The decoder reads both. Set CHORALE_ECC=0 to force plain.
"""
import io
import json
import math
import os
import random
import secrets
import sys
import tempfile
import threading
import wave
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, abort, jsonify, render_template, request, send_file

import renderer
from chorale_engine import (
    CapacityError, ChoraleRejected, DecodeError, capacity_report, decode_musicxml_with_style,
    STYLES, encode_style as engine_encode, load_tune, read_musicxml,
)
from chorale_engine.ecc import decode_words, ecc_capacity, encode_words, pack_indices
from chorale_engine.musicxml_io import musicxml_string
from chorale_engine.performance import performance
from chorale_engine.styles import style_capacity, style_writing

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


# ---------------------------------------------------------- chorale choice
def _pool_path():
    from chorale_engine.corpus import _cache_dir
    from chorale_engine.options import _source_digest
    return _cache_dir() / f"webapp-capacity-{_source_digest()}.json"


_POOL = None
_POOL_LOCK = threading.Lock()


def chorale_pool():
    """{style: {chorale id: capacity}} for the usable chorales (one per tune), worked out once and saved.

    It is the same for every message, so it is kept on disk next to the engine's own cache,
    under a name that changes whenever the engine's code does.
    """
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        path = _pool_path()
        try:
            saved = json.loads(path.read_text())
            _POOL = {style: {int(c): int(v) for c, v in caps.items()} for style, caps in saved.items()}
            return _POOL
        except (OSError, ValueError):
            pass
        #one chorale per tune: twins with the same soprano line are told apart by decoding, which
        #reports the lower number, so the page would name the wrong one
        from chorale_engine.identify import _signatures
        seen, distinct = set(), []
        for cid, end, beats in _signatures():
            if (end, beats) not in seen:
                seen.add((end, beats))
                distinct.append(cid)
        pool = {}
        for style in STYLES:
            caps = {}
            for cid in distinct:
                try:
                    caps[cid] = style_capacity(cid, style)["capacity"]
                except ChoraleRejected:
                    pass  # some chorales cannot be written in the fancy style
            pool[style] = caps
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({s: {str(c): str(v) for c, v in caps.items()} for s, caps in pool.items()}))
        except OSError:
            pass
        _POOL = pool
        return _POOL


def chorales_that_fit(n, style):
    """Chorale IDs that can hold the integer n in the style, ascending."""
    return sorted(cid for cid, cap in chorale_pool()[style].items() if n < cap)


def pick_chorale(n, style):
    """A chorale chosen at random from those that can hold n, so messages do not all sound the same."""
    fits = chorales_that_fit(n, style)
    if not fits:
        raise ValueError("That message is too long for any chorale in this style.")
    return random.SystemRandom().choice(fits)


def chorale_name(cid):
    try:
        title = load_tune(cid).title.strip()
    except Exception:
        title = ""
    return f"Chorale {cid}" + (f": {title}" if title else "")


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


def encode_indices(indices, style="original"):
    """(chorale id, chords) for the message."""
    kind, r = mode()
    if kind == "ecc":
        return CHORALE_ID, encode_words(CHORALE_ID, indices, len(WORDS), r)
    if any(i + 10000 > 99999 for i in indices):
        raise ValueError("The word list is too long for plain mode.")
    n = int("".join(str(i + 10000) for i in indices))
    cid = pick_chorale(n, style)
    return cid, engine_encode(cid, n, style)


def message_integer(indices):
    """The integer the message is turned into before it goes into the music, in the active mode."""
    kind, _r = mode()
    if kind == "ecc":
        return pack_indices(indices, len(WORDS))
    return int("".join(str(i + 10000) for i in indices))


def payload_bits(indices):
    """How many bits the message really takes: the size of that integer, not the length of the text."""
    return message_integer(indices).bit_length()


def capacity_bits(style="original"):
    """The most payload bits any chorale has room for in the active mode and style."""
    kind, r = mode()
    if kind == "ecc":
        return ecc_capacity(CHORALE_ID, r)["bits"]
    return max(chorale_pool()[style].values()).bit_length() - 1


def decode_file(path):
    """(word indices, chords or None, repaired phrase indices, chorale id, style)."""
    kind, r = mode()
    ecc_error = None
    if kind == "ecc":
        try:
            idx, repaired, bad = decode_words(CHORALE_ID, read_musicxml(CHORALE_ID, path), len(WORDS), r)
            if 1 <= len(idx) <= MAX_WORDS:
                return idx, repaired, bad, CHORALE_ID, "original"
        except (DecodeError, ValueError) as exc:
            ecc_error = exc
    try:
        cid, n, style = decode_musicxml_with_style(path)
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
    return idx, chords, [], cid, style


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


def chords_to_wav(chords, cid=CHORALE_ID, style="original", bpm=76, rate=16000):
    """A small additive-synth rendering of the score as WAV bytes: every note as written, so the
    soprano keeps Bach's rhythm and the fancy style is heard with its walking bass, passing notes
    and ornaments."""
    import numpy

    voices = performance(cid, chords, style=style)
    per_quarter = 60.0 / bpm * rate
    end = max(float(off + dur) for notes in voices for off, dur, _m in notes)
    mix = numpy.zeros(int(end * per_quarter) + rate // 2, numpy.float32)
    gains = (1.0, 0.8, 0.8, 0.9)  # the tune a little louder than the inner voices
    for gain, notes in zip(gains, voices):
        for off, dur, midi in notes:
            start, n = int(float(off) * per_quarter), max(1, int(float(dur) * per_quarter))
            t = numpy.arange(n, dtype=numpy.float32)
            w = 2 * math.pi * 440.0 * 2 ** ((midi - 69) / 12) / rate
            atk, rel = max(1, min(int(0.01 * rate), n // 4)), max(1, min(int(0.06 * rate), n // 3))
            env = numpy.ones(n, numpy.float32)
            env[:atk] = numpy.linspace(0, 1, atk, endpoint=False)
            env[n - rel:] = numpy.linspace(1, 0, rel)
            tone = numpy.sin(w * t) + 0.4 * numpy.sin(2 * w * t) + 0.15 * numpy.sin(3 * w * t)
            mix[start:start + n] += gain * env * tone
    peak = float(numpy.abs(mix).max()) or 1.0
    pcm = (mix * (0.85 * 32767 / max(peak, 4.0))).astype("<i2")
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


@app.route("/payload")
def payload():
    """Live payload size for the message typed so far: {"bits", "capacity", "words", "error"}."""
    raw = request.args.get("message", "")
    style = request.args.get("style", "original")
    if style not in STYLES:
        style = "original"
    out = {"bits": 0, "capacity": capacity_bits(style), "chorales": len(chorale_pool()[style]),
           "words": len(raw.split()), "error": None}
    if raw.split():
        try:
            indices = words_to_indices(raw)
            out["bits"] = payload_bits(indices)
            if mode()[0] == "ecc":
                out["chorales"] = 1 if out["bits"] <= out["capacity"] else 0
            else:
                out["chorales"] = len(chorales_that_fit(message_integer(indices), style))
            if not out["chorales"]:
                out["error"] = f"That message needs {out['bits']} bits; no chorale holds that many in this style."
        except ValueError as exc:
            out["error"] = str(exc)
    return jsonify(out)


@app.route("/encode", methods=["POST"])
def encode():
    encoding_style = request.form.get("encoding_style", "original")
    try:
        if encoding_style not in STYLES:
            raise ValueError(f"Choose an encoding style: {', '.join(STYLES)}.")
        cid, chords = encode_indices(words_to_indices(request.form.get("secret_message", "")), encoding_style)
        xml = musicxml_string(cid, chords, **style_writing(encoding_style))
    except USER_ERRORS as exc:
        return render_template(
            "index.html", active_tab="encode", error=str(exc),
            encoding_style=encoding_style if encoding_style in STYLES else "original",
        )
    png = optional("PNG", lambda: renderer.writeScore(xml, None, "png", 2000, 1.0))
    wav = optional("WAV", lambda: chords_to_wav(chords, cid, encoding_style))
    return render_template(
        "index.html", active_tab="encode", chorale_name=chorale_name(cid),
        xml_file=stash(xml, "musicxml", "application/vnd.recordare.musicxml+xml"),
        png_file=stash(png, "png", "image/png") if png else None,
        wav_file=stash(wav, "wav", "audio/wav") if wav else None,
        encoding_style=encoding_style,
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
            indices, chords, bad, cid, style = decode_file(str(path))
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="decode", error=str(exc))
    wav = optional("WAV", lambda: chords_to_wav(chords, cid, style)) if chords else None
    note = (f"Repaired damaged phrase(s) {', '.join(str(p + 1) for p in bad)}. The message is intact."
            if bad else "No damage found.")
    return render_template("index.html", active_tab="decode", chorale_name=chorale_name(cid),
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


# work out the chorale capacities in the background, so the first message does not wait for them
threading.Thread(target=chorale_pool, daemon=True).start()


if __name__ == "__main__":
    app.run(debug=True)
