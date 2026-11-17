"""Corpus loading, filtering, beats, phrases, fermatas and Bach's opening chord.

Chorales are addressed by Riemenschneider number through music21's own
numbering (corpus.chorales). Nothing is hardcoded here.

Parsing and key analysis are slow, so the raw extraction of each chorale is
cached as JSON under .cache/ in the project root, keyed by the music21
version. Everything derived from it (filter, beats, key map) is recomputed
on load.
"""

from __future__ import annotations

import json
import os
import pickle
import threading
import zlib
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import music21

from .errors import ChoraleRejected
from .keys import Key, NoKeyMap, home_key_from_score, home_tonic_pcs, key_map

MUSIC21_VERSION = music21.__version__
_RAW_SCHEMA = 1
ALLOWED_METERS = ("4/4",)
# Two groups of chorales are rejected for now and may be brought back later.
REJECT_FINAL_OUTSIDE_TONIC = True
REJECT_KEY_MAP_WITHOUT_HOME = True

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _cache_dir() -> Path:
    base = os.environ.get("CHORALE_ENGINE_CACHE")
    root = Path(base) if base else _PROJECT_ROOT / ".cache"
    return root / f"music21-{MUSIC21_VERSION}-raw{_RAW_SCHEMA}"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SopranoEvent:
    """One soprano note or rest exactly as written in the corpus file."""

    offset: Fraction
    length: Fraction
    midi: int | None  # None for a rest
    step: str | None
    alter: int | None
    octave: int | None
    fermata: bool
    tie: str | None  # 'start', 'continue', 'stop' or None


@dataclass(frozen=True)
class Measure:
    offset: Fraction
    length: Fraction
    number: int


@dataclass(frozen=True)
class Beat:
    """One decision point: a crotchet beat where the soprano sounds."""

    index: int  # position in the whole piece
    offset: int  # quarter-length offset from the start of the piece
    soprano_midi: int
    phrase_index: int
    beat_index: int  # position inside the phrase
    is_cadence: bool
    length: Fraction  # how long this beat's chord sounds, in quarter lengths
    key: Key


@dataclass(frozen=True)
class Tune:
    chorale_id: int
    title: str
    home_key: Key
    beats: tuple[Beat, ...]
    phrases: tuple[tuple[int, int], ...]  # (first beat index, end beat index) per phrase
    key_map: tuple[Key, ...]  # one key per beat, same order as beats
    soprano: tuple[SopranoEvent, ...]
    measures: tuple[Measure, ...]
    key_signature: int  # sharps (negative for flats) in the corpus file
    bach_opening: tuple[int, int, int, int] | None  # Bach's (S, A, T, B) at the first beat

    @property
    def n_phrases(self) -> int:
        return len(self.phrases)

    def phrase_beats(self, phrase_index: int) -> tuple[Beat, ...]:
        start, end = self.phrases[phrase_index]
        return self.beats[start:end]

    def beat(self, phrase_index: int, beat_index: int) -> Beat:
        start, end = self.phrases[phrase_index]
        if not 0 <= beat_index < end - start:
            raise IndexError(f"phrase {phrase_index} has no beat {beat_index}")
        return self.beats[start + beat_index]


# ---------------------------------------------------------------------------
# Raw extraction (the only part that touches music21 scores)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def chorale_ids() -> tuple[int, ...]:
    """Every Riemenschneider number music21 knows, ascending."""
    table = music21.corpus.chorales.ChoraleListRKBWV().byRiemenschneider
    return tuple(sorted(int(k) for k in table))


# music21 keeps each parsed chorale as a pickle in its temp folder. Two parses of the same chorale at once
# (the web app warms its caches on several threads, and Flask's debug reloader starts a second process) can
# leave a half-written pickle behind, which then fails to load with "incomplete or truncated stream".
_PARSE_LOCK = threading.Lock()


def _clear_music21_cache() -> None:
    """Delete music21's cached pickles, so the next parse reads the original corpus files again."""
    root = Path(music21.environment.Environment().getRootTempDir())
    for path in root.glob("m21-*.p*"):
        try:
            path.unlink()
        except OSError:
            pass


def parse_score(chorale_id: int):
    """Parse Bach's full score for a Riemenschneider number from the corpus."""
    with _PARSE_LOCK:
        for attempt in (0, 1):
            try:
                it = music21.corpus.chorales.Iterator(
                    chorale_id, chorale_id, numberingSystem="riemenschneider"
                )
                scores = list(it)
                break
            except (zlib.error, EOFError, pickle.UnpicklingError):
                if attempt:  # a fresh parse failed too: a real problem, not a damaged cache
                    raise
                _clear_music21_cache()
    if len(scores) != 1:
        raise ChoraleRejected(chorale_id, "not found in the music21 corpus")
    return scores[0]


def _frac(x) -> Fraction:
    return Fraction(x).limit_denominator(1 << 16)


def _fs(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _has_fermata(el) -> bool:
    return any(isinstance(e, music21.expressions.Fermata) for e in el.expressions)


def _sounding_midi(part_flat, offset) -> int | None:
    """MIDI of the single note sounding at an offset in a flat part, else None."""
    found = None
    for el in part_flat.notesAndRests:
        start = _frac(el.offset)
        length = _frac(el.duration.quarterLength)
        if length == 0:
            continue
        if start <= offset < start + length:
            if el.isRest or el.isChord or found is not None:
                return None
            found = int(el.pitch.midi)
    return found


def extract_raw(chorale_id: int) -> dict:
    """Read everything the engine needs from the corpus file, as plain JSON data."""
    score = parse_score(chorale_id)
    parts = list(score.parts)
    raw: dict = {
        "chorale_id": chorale_id,
        "title": score.metadata.title if score.metadata is not None else "",
        "n_parts": len(parts),
        "meters": sorted(
            {ts.ratioString for ts in score.recurse().getElementsByClass(music21.meter.TimeSignature)}
        ),
    }
    home = home_key_from_score(score)
    raw["home_key"] = [home.tonic, home.mode] if home is not None else None
    if not parts:
        return raw

    sop = parts[0]
    flat = sop.flatten()
    events = []
    monophonic = True
    for el in flat.notesAndRests:
        length = _frac(el.duration.quarterLength)
        if length == 0:
            continue  # grace note
        ev = {
            "offset": _fs(_frac(el.offset)),
            "length": _fs(length),
            "midi": None,
            "step": None,
            "alter": None,
            "octave": None,
            "fermata": _has_fermata(el),
            "tie": el.tie.type if el.tie is not None else None,
        }
        if el.isChord:
            monophonic = False
        elif el.isNote:
            ev["midi"] = int(el.pitch.midi)
            ev["step"] = el.pitch.step
            ev["alter"] = int(el.pitch.alter)
            ev["octave"] = int(el.pitch.implicitOctave)
        events.append(ev)
    if list(sop.recurse().getElementsByClass(music21.stream.Voice)):
        monophonic = False
    raw["soprano_monophonic"] = monophonic
    raw["soprano"] = events
    raw["measures"] = [
        {
            "offset": _fs(_frac(m.offset)),
            "length": _fs(_frac(m.duration.quarterLength)),
            "number": int(m.number),
        }
        for m in sop.getElementsByClass(music21.stream.Measure)
    ]
    ks = sop.recurse().getElementsByClass(music21.key.KeySignature).first()
    raw["key_signature"] = int(ks.sharps) if ks is not None else 0

    opening = None
    first = next((e for e in events if e["midi"] is not None), None)
    if first is not None and len(parts) == 4:
        at = Fraction(first["offset"])
        opening = [_sounding_midi(p.flatten(), at) for p in parts]
        if any(v is None for v in opening):
            opening = None
    raw["bach_opening"] = opening
    return raw


@lru_cache(maxsize=None)
def load_raw(chorale_id: int) -> dict:
    """Raw extraction for a chorale, from the JSON cache when present."""
    if chorale_id not in chorale_ids():
        raise ChoraleRejected(chorale_id, "not a Riemenschneider number in the music21 corpus")
    path = _cache_dir() / f"chorale_{chorale_id:03d}.json"
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            pass
    raw = extract_raw(chorale_id)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(raw, fh, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass
    return raw


# ---------------------------------------------------------------------------
# Beats, phrases, filter
# ---------------------------------------------------------------------------

def _soprano_events(raw: dict) -> tuple[SopranoEvent, ...]:
    out = [
        SopranoEvent(
            offset=Fraction(e["offset"]),
            length=Fraction(e["length"]),
            midi=e["midi"],
            step=e["step"],
            alter=e["alter"],
            octave=e["octave"],
            fermata=bool(e["fermata"]),
            tie=e["tie"],
        )
        for e in raw["soprano"]
    ]
    out.sort(key=lambda e: e.offset)
    return tuple(out)


def _written_notes(events) -> list[list]:
    """Soprano notes as written: [offset, length, midi, fermata, tied_to_next]."""
    return [
        [e.offset, e.length, e.midi, e.fermata, e.tie in ("start", "continue")]
        for e in events
        if e.midi is not None
    ]


def _cadence_spans(notes) -> list[tuple[Fraction, Fraction]]:
    """(start, end) of every cadence note.

    A cadence starts at the written note that carries the fermata. Notes tied
    on after it belong to the same held chord. Notes tied before it are
    ordinary beats, so a tied soprano note can carry more than one chord.
    """
    spans: list[tuple[Fraction, Fraction]] = []
    for i, (off, length, midi, fermata, _tied) in enumerate(notes):
        if not fermata or (spans and off < spans[-1][1]):
            continue
        end = off + length
        j = i
        while (
            notes[j][4]
            and j + 1 < len(notes)
            and notes[j + 1][2] == midi
            and notes[j + 1][0] == end
        ):
            j += 1
            end += notes[j][1]
        spans.append((off, end))
    return spans


def _build(raw: dict):
    """Return (Tune, None) or (None, rejection reason)."""
    cid = raw["chorale_id"]
    if raw["n_parts"] != 4:
        return None, f"not four parts ({raw['n_parts']} parts)"
    if tuple(raw["meters"]) != ALLOWED_METERS:
        return None, f"meter not 4/4 ({', '.join(raw['meters']) or 'none'})"
    if not raw["soprano_monophonic"]:
        return None, "soprano is not a single line"
    events = _soprano_events(raw)
    notes = _written_notes(events)
    if not notes:
        return None, "soprano has no notes"
    if not any(n[3] for n in notes):
        return None, "no soprano fermatas"
    measures = tuple(
        Measure(Fraction(m["offset"]), Fraction(m["length"]), m["number"]) for m in raw["measures"]
    )
    if any(m.offset.denominator != 1 for m in measures):
        return None, "barlines are off the crotchet grid"

    spans = _cadence_spans(notes)
    end = max(n[0] + n[1] for n in notes)
    if spans[-1][1] != end:
        # The final note is a cadence even without a fermata.
        notes[-1][3] = True
        spans = _cadence_spans(notes)
    if any(start.denominator != 1 for start, _end in spans):
        return None, "a fermata note starts off the beat"

    slots = []  # (offset, midi, is_cadence, length)
    ni = 0
    si = 0
    t = 0
    while t < end:
        while ni < len(notes) and notes[ni][0] + notes[ni][1] <= t:
            ni += 1
        while si < len(spans) and spans[si][1] <= t:
            si += 1
        if ni < len(notes) and notes[ni][0] <= t:
            midi = notes[ni][2]
            if si < len(spans) and spans[si][0] <= t:
                if spans[si][0] == t:
                    slots.append((t, midi, True, spans[si][1] - spans[si][0]))
                # later beats of a held cadence note are not separate beats
            else:
                slots.append((t, midi, False, Fraction(1)))
        t += 1
    if not slots or not slots[-1][2]:
        return None, "the final soprano note does not start on a beat"

    home = Key(raw["home_key"][0], raw["home_key"][1])
    try:
        kmap = key_map([s[1] % 12 for s in slots], home)
    except NoKeyMap as exc:
        return None, f"no key map: {exc}"
    # Deferred groups: rejected for now, to be revisited at the end.
    if REJECT_FINAL_OUTSIDE_TONIC and slots[-1][1] % 12 not in home_tonic_pcs(home):
        return None, "the final soprano note is not in the home tonic triad"
    if REJECT_KEY_MAP_WITHOUT_HOME and all(k != home for k in kmap):
        return None, "the key map never uses the home key"

    beats = []
    phrases = []
    start = 0
    for i, (off, midi, cad, length) in enumerate(slots):
        beats.append(
            Beat(
                index=i,
                offset=int(off),
                soprano_midi=midi,
                phrase_index=len(phrases),
                beat_index=i - start,
                is_cadence=cad,
                length=length,
                key=kmap[i],
            )
        )
        if cad:
            phrases.append((start, i + 1))
            start = i + 1
    if phrases[0][1] - phrases[0][0] < 2:
        return None, "the first phrase has a single beat"

    opening = raw.get("bach_opening")
    tune = Tune(
        chorale_id=cid,
        title=raw["title"] or "",
        home_key=home,
        beats=tuple(beats),
        phrases=tuple(phrases),
        key_map=tuple(kmap),
        soprano=events,
        measures=measures,
        key_signature=raw["key_signature"],
        bach_opening=tuple(opening) if opening else None,
    )
    return tune, None


@lru_cache(maxsize=None)
def _built(chorale_id: int):
    return _build(load_raw(chorale_id))


def filter_reason(chorale_id: int) -> str | None:
    """Why the corpus filter rejects this chorale, or None if it passes."""
    try:
        return _built(chorale_id)[1]
    except ChoraleRejected as exc:
        return exc.reason


def load_tune(chorale_id: int) -> Tune:
    """Beats, phrases and key map for a chorale. Raises ChoraleRejected."""
    tune, reason = _built(chorale_id)
    if tune is None:
        raise ChoraleRejected(chorale_id, reason)
    return tune


def filter_usable() -> list[int]:
    """Chorale IDs that pass the corpus filter, ascending."""
    return [cid for cid in chorale_ids() if filter_reason(cid) is None]


def filter_rejected() -> dict[int, str]:
    """Chorale IDs the corpus filter rejects, with the reason."""
    out = {}
    for cid in chorale_ids():
        reason = filter_reason(cid)
        if reason is not None:
            out[cid] = reason
    return out
