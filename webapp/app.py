"""Web front end: word-list or printable-ASCII messages -> Bach chorales -> MusicXML
(plus PNG and WAV outputs), and back from MusicXML or an ASCII ZIP.

In plain mode each message gets a chorale picked at random from those that can hold it
in the chosen style; the decoder recognises the chorale from its soprano line. In
error-corrected mode the chorale stays fixed (CHORALE_ID).

Checksum (on unless CHORALE_CHECKSUM=0): in plain mode the words are packed as base-30001
digits and a 16-bit checksum is stored with them, so the decoder can repair up to two
misread chords (chorale_engine/checksum.py). Scores made without it still decode.

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
import re
import secrets
import sys
import tempfile
import threading
import wave
import zipfile
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
from chorale_engine.checksum import CHECK_BITS, with_check
from chorale_engine.ecc import decode_words, ecc_capacity, encode_words, pack_indices, unpack_indices
from chorale_engine.identify import decode_score
from chorale_engine.musicxml_io import musicxml_string
from chorale_engine.performance import performance
from chorale_engine.styles import style_capacity, style_writing

CHORALE_ID = 215
MAX_WORDS = 3
CHECKSUM = os.environ.get("CHORALE_CHECKSUM", "1") != "0"
MAX_PARITY = 2
ASCII_MAGIC = (127 << 21) | (126 << 14) | (125 << 7) | 124
ASCII_MAGIC_BITS = 28
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


def checked():
    """Whether messages carry the checksum: plain mode with CHECKSUM on."""
    return CHECKSUM and mode()[0] != "ecc"


def encode_indices(indices, style="original"):
    """(chorale id, chords) for the message."""
    kind, r = mode()
    if kind == "ecc":
        return CHORALE_ID, encode_words(CHORALE_ID, indices, len(WORDS), r)
    if not checked() and any(i + 10000 > 99999 for i in indices):
        raise ValueError("The word list is too long for plain mode.")
    n = message_integer(indices)
    cid = pick_chorale(stored_integer(indices), style)
    return cid, engine_encode(cid, n, style, checksum=True) if checked() else engine_encode(cid, n, style)


def message_integer(indices):
    """The integer the message is turned into before it goes into the music, in the active mode.

    Error-corrected and checksum modes pack the words as base-30001 digits (about 15 bits a
    word); the old plain mode writes five decimal digits per word (about 17 bits a word)."""
    kind, _r = mode()
    if kind == "ecc" or checked():
        return pack_indices(indices, len(WORDS))
    return int("".join(str(i + 10000) for i in indices))


def stored_integer(indices):
    """The number that actually goes into the music: the message, plus its checksum if on."""
    n = message_integer(indices)
    return with_check(n) if checked() else n


def check_bits():
    return CHECK_BITS if checked() else 0


def payload_bits(indices):
    """How many bits the message really takes: the size of that integer, not the length of the text."""
    return message_integer(indices).bit_length()


def validate_ascii_message(raw):
    if not raw:
        raise ValueError("Type a message first.")
    if any(not 32 <= ord(char) <= 126 for char in raw):
        raise ValueError("Use printable ASCII characters only (spaces and punctuation are allowed).")


def pack_ascii_chunk(text):
    """Pack printable ASCII into a self-identifying integer for one chorale."""
    validate_ascii_message(text)
    payload = 0
    for char in text:
        payload = (payload << 7) | ord(char)
    return (ASCII_MAGIC << (7 * len(text))) | payload


def unpack_ascii_chunk(n):
    """Recover a printable ASCII chunk packed by pack_ascii_chunk."""
    if n < 0 or n.bit_length() < ASCII_MAGIC_BITS:
        raise DecodeError("No printable-ASCII message was found in that file.")
    payload_bits = n.bit_length() - ASCII_MAGIC_BITS
    if payload_bits % 7:
        raise DecodeError("No printable-ASCII message was found in that file.")
    length = payload_bits // 7
    if not length or n >> payload_bits != ASCII_MAGIC:
        raise DecodeError("No printable-ASCII message was found in that file.")
    chars = []
    for _ in range(length):
        code = n & 0x7F
        if not 32 <= code <= 126:
            raise DecodeError("No printable-ASCII message was found in that file.")
        chars.append(chr(code))
        n >>= 7
    return "".join(reversed(chars))


def ascii_chunk_length(style):
    """Largest fixed chunk size that fits the highest-capacity chorale in this style."""
    capacities = chorale_pool()[style]
    if not capacities:
        raise ValueError("No chorales are available in this style.")
    capacity = max(capacities.values())
    length = 0
    # the checksum, if on, takes its 16 bits of every part's room
    stored = with_check if checked() else (lambda n: n)
    while stored(pack_ascii_chunk("~" * (length + 1))) < capacity:
        length += 1
    if not length:
        raise ValueError("No chorale has enough capacity for printable-ASCII encoding in this style.")
    return length


def encode_ascii(raw, style="original"):
    """Return (chorale ID, chords) for each fixed-size printable-ASCII chunk."""
    validate_ascii_message(raw)
    chunk_length = ascii_chunk_length(style)
    encoded = []
    for start in range(0, len(raw), chunk_length):
        n = pack_ascii_chunk(raw[start:start + chunk_length])
        cid = pick_chorale(with_check(n) if checked() else n, style)
        encoded.append((cid, engine_encode(cid, n, style, checksum=True) if checked() else engine_encode(cid, n, style)))
    return encoded


def capacity_bits(style="original"):
    """The most payload bits any chorale has room for in the active mode and style."""
    kind, r = mode()
    if kind == "ecc":
        return ecc_capacity(CHORALE_ID, r)["bits"]
    return max(chorale_pool()[style].values()).bit_length() - 1


def decode_ascii_archive(data):
    """Decode an ordered ZIP archive of printable-ASCII MusicXML chunks: (message, chorale ids)."""
    message, chorales, _repairs = decode_ascii_archive_report(data)
    return message, chorales


def decode_ascii_archive_report(data):
    """(message, chorale ids, repairs as text) from a ZIP of printable-ASCII MusicXML chunks."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("Upload a valid ZIP of printable-ASCII MusicXML parts.") from exc
    with archive:
        parts = []
        total_size = 0
        for info in archive.infolist():
            if info.is_dir():
                continue
            match = re.fullmatch(r"part-(\d+)\.musicxml", Path(info.filename).name)
            if not match or Path(info.filename).name != info.filename:
                raise ValueError("The ZIP must contain only numbered part-XXXX.musicxml files.")
            total_size += info.file_size
            if total_size > app.config["MAX_CONTENT_LENGTH"]:
                raise ValueError("The uncompressed MusicXML parts are too large.")
            parts.append((int(match.group(1)), info))
        parts.sort(key=lambda part: part[0])
        if not parts or [number for number, _info in parts] != list(range(1, len(parts) + 1)):
            raise ValueError("The ZIP must contain consecutively numbered MusicXML parts.")

        decoded, chorales, repairs = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            for index, (_number, info) in enumerate(parts):
                try:
                    content = archive.read(info)
                except (RuntimeError, OSError, zipfile.BadZipFile) as exc:
                    raise ValueError("Could not read every MusicXML part from the ZIP.") from exc
                path = Path(tmp) / f"part-{index + 1}.musicxml"
                path.write_bytes(content)
                text, cid, fixed = _decode_ascii_part(str(path))
                decoded.append(text)
                chorales.append(cid)
                repairs += [f"part {index + 1}, phrase {p + 1}, beat {b + 1}" for p, b in fixed]
    return "".join(decoded), chorales, repairs


def _decode_ascii_part(path):
    """(text, chorale id, repaired (phrase, beat)) for one ASCII part: checksum format as read,
    then the format without a checksum, then the checksum format with repair."""
    def checked_reading(max_errors):
        reading = decode_score(path, checksum=True, max_errors=max_errors)
        return unpack_ascii_chunk(reading.n), reading.chorale_id, reading.repaired

    def plain_reading():
        cid, n, _style = decode_musicxml_with_style(path)
        return unpack_ascii_chunk(n), cid, []

    for attempt in (lambda: checked_reading(0), plain_reading, lambda: checked_reading(None)):
        try:
            return attempt()
        except (DecodeError, ValueError, CapacityError):
            continue
    raise DecodeError("A chorale part could not be read, or is too damaged to repair.")


def _legacy_words(n):
    """Word indices from the old plain format (five decimal digits a word), or None."""
    digits = str(n)
    idx = [int(digits[i:i + 5]) - 10000 for i in range(0, len(digits), 5)]
    if len(digits) % 5 or not 1 <= len(idx) <= MAX_WORDS or any(not 0 <= i < len(WORDS) for i in idx):
        return None
    return idx


def _packed_words(n):
    """Word indices from base-30001 packing, or None."""
    try:
        idx = unpack_indices(n, len(WORDS))
    except ValueError:
        return None
    return idx if 1 <= len(idx) <= MAX_WORDS and all(0 <= i < len(WORDS) for i in idx) else None


def decode_file(path):
    """(word indices, chords as written, repairs (text), chorale id, style).

    Tries, in order: the checksum format as read; the old plain format; then the checksum
    format with repair of up to two misread chords. A clean reading is always preferred to a
    repaired one, so a score from either format decodes as it was written.
    """
    kind, r = mode()
    ecc_error = None
    if kind == "ecc":
        try:
            idx, repaired, bad = decode_words(CHORALE_ID, read_musicxml(CHORALE_ID, path), len(WORDS), r)
            if 1 <= len(idx) <= MAX_WORDS:
                return idx, repaired, [f"phrase {p + 1}" for p in bad], CHORALE_ID, "original"
        except (DecodeError, ValueError) as exc:
            ecc_error = exc

    def checked_reading(max_errors):
        reading = decode_score(path, checksum=True, max_errors=max_errors)
        idx = _packed_words(reading.n)
        if idx is None:
            raise DecodeError("No message from this tool was found in that file.")
        chords = engine_encode(reading.chorale_id, reading.n, reading.style, checksum=True)
        repairs = [f"phrase {p + 1}, beat {b + 1}" for p, b in reading.repaired]
        return idx, chords, repairs, reading.chorale_id, reading.style

    def plain_reading():
        reading = decode_score(path)
        idx = _legacy_words(reading.n)
        if idx is None:
            raise DecodeError("No message from this tool was found in that file.")
        chords = engine_encode(reading.chorale_id, reading.n, reading.style)
        return idx, chords, [], reading.chorale_id, reading.style

    for attempt in (lambda: checked_reading(0), plain_reading, lambda: checked_reading(None)):
        try:
            return attempt()
        except (DecodeError, ValueError, CapacityError):
            continue
    raise ecc_error or DecodeError("No message from this tool was found in that file, "
                                   "or it is too damaged to repair.")


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


def zip_files(files):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in files:
            bundle.writestr(name, data)
    return archive.getvalue()


def combine_pngs(images):
    """Stack rendered chorale PNGs vertically into one image."""
    from PIL import Image

    decoded = []
    for data in images:
        with Image.open(io.BytesIO(data)) as image:
            decoded.append(image.convert("RGB"))
    width = max(image.width for image in decoded)
    height = sum(image.height for image in decoded)
    combined = Image.new("RGB", (width, height), "white")
    top = 0
    for image in decoded:
        combined.paste(image, ((width - image.width) // 2, top))
        top += image.height
        image.close()
    out = io.BytesIO()
    combined.save(out, format="PNG")
    combined.close()
    return out.getvalue()


def merge_wav_files(files):
    """Concatenate compatible WAV files in chorale order."""
    if not files:
        raise ValueError("No WAV parts to merge.")
    out = io.BytesIO()
    with wave.open(out, "wb") as merged:
        expected = None
        for data in files:
            with wave.open(io.BytesIO(data), "rb") as part:
                params = part.getparams()
                signature = (params.nchannels, params.sampwidth, params.framerate,
                             params.comptype, params.compname)
                if expected is None:
                    expected = signature
                    merged.setparams(params)
                elif signature != expected:
                    raise ValueError("WAV parts have incompatible audio formats.")
                merged.writeframes(part.readframes(params.nframes))
    return out.getvalue()


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
    """Live payload size for the message typed so far:
    {"bits", "check_bits", "capacity", "chorales", "words", "error"}."""
    raw = request.args.get("message", "")
    style = request.args.get("style", "original")
    method = request.args.get("method", "words")
    if style not in STYLES:
        style = "original"
    out = {"bits": 0, "check_bits": check_bits(), "capacity": capacity_bits(style),
           "chorales": len(chorale_pool()[style]), "words": len(raw.split()), "error": None}
    if raw:
        if method == "ascii":
            try:
                validate_ascii_message(raw)
                out["capacity"] = max(chorale_pool()[style].values()).bit_length() - 1
                out["chorales"] = math.ceil(len(raw) / ascii_chunk_length(style))
                out["bits"] = len(raw) * 7 + out["chorales"] * ASCII_MAGIC_BITS
                out["check_bits"] = out["chorales"] * check_bits()
            except ValueError as exc:
                out["error"] = str(exc)
        elif raw.split():
            try:
                indices = words_to_indices(raw)
                out["bits"] = payload_bits(indices)
                if mode()[0] == "ecc":
                    out["chorales"] = 1 if out["bits"] <= out["capacity"] else 0
                else:
                    out["chorales"] = len(chorales_that_fit(stored_integer(indices), style))
                if not out["chorales"]:
                    need = out["bits"] + out["check_bits"]
                    out["error"] = f"That message needs {need} bits; no chorale holds that many in this style."
            except ValueError as exc:
                out["error"] = str(exc)
    return jsonify(out)


@app.route("/encode", methods=["POST"])
def encode():
    encoding_style = request.form.get("encoding_style", "original")
    encoding_method = request.form.get("encoding_method", "words")
    try:
        if encoding_style not in STYLES:
            raise ValueError(f"Choose an encoding style: {', '.join(STYLES)}.")
        message = request.form.get("secret_message", "")
        if encoding_method == "ascii":
            encoded_parts = encode_ascii(message, encoding_style)
            xml_parts, png_parts, wav_parts = [], [], []
            for index, (cid, chords) in enumerate(encoded_parts, start=1):
                xml = musicxml_string(cid, chords, **style_writing(encoding_style))
                part_name = f"part-{index:04d}"
                xml_parts.append((f"{part_name}.musicxml", xml))
                png = optional("PNG", lambda: renderer.writeScore(xml, None, "png", 2000, 1.0))
                wav = optional("WAV", lambda: chords_to_wav(chords, cid, encoding_style))
                if png:
                    png_parts.append(png)
                if wav:
                    wav_parts.append(wav)
            combined_png = (
                optional("Combined PNG", lambda: combine_pngs(png_parts))
                if len(png_parts) == len(encoded_parts) else None
            )
            combined_wav = (
                optional("Combined WAV", lambda: merge_wav_files(wav_parts))
                if len(wav_parts) == len(encoded_parts) else None
            )
            bundle_file = stash(zip_files(xml_parts), "zip", "application/zip")
            return render_template(
                "index.html", active_tab="encode", chorale_name=f"{len(encoded_parts)} chorale part(s)",
                bundle_file=bundle_file, encoding_style=encoding_style, encoding_method=encoding_method,
                png_file=stash(combined_png, "png", "image/png") if combined_png else None,
                wav_file=stash(combined_wav, "wav", "audio/wav") if combined_wav else None,
            )
        if encoding_method != "words":
            raise ValueError("Choose a supported encoding method.")
        cid, chords = encode_indices(words_to_indices(message), encoding_style)
        xml = musicxml_string(cid, chords, **style_writing(encoding_style))
    except USER_ERRORS as exc:
        return render_template(
            "index.html", active_tab="encode", error=str(exc),
            encoding_style=encoding_style if encoding_style in STYLES else "original",
            encoding_method=encoding_method if encoding_method in ("words", "ascii") else "words",
        )
    png = optional("PNG", lambda: renderer.writeScore(xml, None, "png", 2000, 1.0))
    wav = optional("WAV", lambda: chords_to_wav(chords, cid, encoding_style))
    return render_template(
        "index.html", active_tab="encode", chorale_name=chorale_name(cid),
        xml_file=stash(xml, "musicxml", "application/vnd.recordare.musicxml+xml"),
        png_file=stash(png, "png", "image/png") if png else None,
        wav_file=stash(wav, "wav", "audio/wav") if wav else None,
        encoding_style=encoding_style,
        encoding_method=encoding_method,
    )


@app.route("/decode", methods=["POST"])
def decode():
    upload = request.files.get("encoded_file")
    if not upload or not upload.filename:
        return render_template("index.html", active_tab="decode", error="No file provided.")
    try:
        suffix = Path(upload.filename).suffix.lower()
        if suffix == ".zip":
            decoded_result, cids, repairs = decode_ascii_archive_report(upload.read())
            note = f"Decoded {len(cids)} printable-ASCII chorale part(s)."
            if repairs:
                note += f" Repaired {len(repairs)} misread chord(s): {'; '.join(repairs)}. The message is intact."
            return render_template(
                "index.html", active_tab="decode", decoded_result=decoded_result,
                repair_note=note,
                chorale_name=f"{len(cids)} chorale part(s)",
            )
        if suffix not in (".musicxml", ".xml"):
            raise ValueError("Upload a .musicxml, .xml, or printable-ASCII .zip file.")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "score.musicxml"
            path.write_bytes(upload.read())
            indices, chords, bad, cid, style = decode_file(str(path))
    except USER_ERRORS as exc:
        return render_template("index.html", active_tab="decode", error=str(exc))
    wav = optional("WAV", lambda: chords_to_wav(chords, cid, style)) if chords else None
    note = (f"Repaired {len(bad)} misread chord(s): {'; '.join(bad)}. The message is intact."
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
