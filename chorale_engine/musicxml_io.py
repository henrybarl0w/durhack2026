"""MusicXML export and parse back to per-beat (S, A, T, B) tuples.

The score is one part on a grand staff: S (voice 1, stems up) and A
(voice 2, stems down) on the treble staff, T (voice 3, stems up) and B
(voice 4, stems down) on the bass staff. It is written by hand so that it is
byte-identical for the same input: fixed part ID and title, no dates and no
generated IDs.

Every note whose pitch is not the key signature's pitch for that letter
carries its own printed accidental, even when the same note already had one
earlier in the bar, and so does a tied note. Where a letter appears with two
different accidentals in one bar on one staff, every such note gets a sign,
including a sign that only restates the key signature. So each pitch can be
read from its own notehead and sign, with no carrying through the bar.

Reading does not trust voice numbers. At each beat it takes the set of
pitches sounding on each staff. The voices never cross (S >= A, T >= B), so
the higher pitch is S (or T) and the lower is A (or B), and a single
notehead stands for both voices of that staff. This is how a note shared by
two voices is read back.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from math import lcm

from .chords import beat_context, spell
from .corpus import load_tune
from .errors import DecodeError
from .realise import realise

PART_ID = "P1"
VOICE_NAMES = ("S", "A", "T", "B")
# (staff, stem) of S, A, T, B.
_LAYOUT = ((1, "up"), (1, "down"), (2, "up"), (2, "down"))
_FERMATA = ("upright", None, None, "inverted")  # above the treble, below the bass
_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# Note values the writer uses, longest first: (quarter length, type, dots).
_VALUES = [
    (Fraction(4), "whole", 0), (Fraction(3), "half", 1), (Fraction(2), "half", 0),
    (Fraction(3, 2), "quarter", 1), (Fraction(1), "quarter", 0),
    (Fraction(3, 4), "eighth", 1), (Fraction(1, 2), "eighth", 0),
    (Fraction(3, 8), "16th", 1), (Fraction(1, 4), "16th", 0),
    (Fraction(1, 8), "32nd", 0), (Fraction(1, 16), "64th", 0),
]


def _pieces(length: Fraction):
    """Split a length into writable note values."""
    out = []
    for value, name, dots in _VALUES:
        while length >= value:
            out.append((value, name, dots))
            length -= value
    if length:
        raise ValueError(f"cannot notate a remainder of {length} quarter lengths")
    return out


def _segments(tune, chords):
    """Per part: [offset, length, pitch or None, fermata, tie_in, tie_out] covering the piece."""
    end = tune.measures[-1].offset + tune.measures[-1].length
    spellings = [spell(beat_context(tune, b.index), c) for b, c in zip(tune.beats, chords)]

    # Soprano: the original rhythm. Notes that sound on a beat take the key spelling.
    sop_spelling = {}
    for beat, sp in zip(tune.beats, spellings):
        for i, e in enumerate(tune.soprano):
            if e.midi is not None and e.offset <= beat.offset < e.offset + e.length:
                sop_spelling.setdefault(i, sp[0])
    soprano = []
    for i, e in enumerate(tune.soprano):
        pitch = None
        if e.midi is not None:
            pitch = sop_spelling.get(i, (e.step, e.alter, e.octave))
        soprano.append([e.offset, e.length, pitch, e.fermata and pitch is not None,
                        e.tie in ("continue", "stop"), e.tie in ("start", "continue")])
    parts = [soprano]

    # A, T, B: one note per beat, held for the beat's length, rests elsewhere.
    for v in (1, 2, 3):
        segs = []
        at = Fraction(0)
        for beat, sp in zip(tune.beats, spellings):
            if beat.offset > at:
                segs.append([at, beat.offset - at, None, False, False, False])
            segs.append([Fraction(beat.offset), beat.length, sp[v], beat.is_cadence, False, False])
            at = beat.offset + beat.length
        if at < end:
            segs.append([at, end - at, None, False, False, False])
        parts.append(segs)
    return parts


_ACCIDENTAL_NAMES = {-2: "flat-flat", -1: "flat", 0: "natural", 1: "sharp", 2: "double-sharp"}


def _key_alters(fifths: int) -> dict[str, int]:
    """Alteration of each letter in a key signature."""
    alters = {step: 0 for step in "CDEFGAB"}
    if fifths > 0:
        for step in "FCGDAEB"[:fifths]:
            alters[step] = 1
    else:
        for step in "BEADGCF"[:-fifths]:
            alters[step] = -1
    return alters


def _accidental_rule(tune, parts):
    """A function (voice, measure index, pitch) -> printed accidental name or None."""
    key = _key_alters(tune.key_signature)
    starts = [m.offset for m in tune.measures]
    ends = [m.offset + m.length for m in tune.measures]
    seen: dict[tuple, set[int]] = {}  # (staff, measure, step, octave) -> alters in that bar
    for v, segs in enumerate(parts):
        staff = _LAYOUT[v][0]
        for off, length, pitch, *_rest in segs:
            if pitch is None:
                continue
            step, alter, octave = pitch
            for mi, (lo, hi) in enumerate(zip(starts, ends)):
                if off < hi and off + length > lo:
                    seen.setdefault((staff, mi, step, octave), set()).add(alter)

    def rule(voice, mi, pitch):
        step, alter, octave = pitch
        clash = len(seen.get((_LAYOUT[voice][0], mi, step, octave), ())) > 1
        if alter != key[step] or clash:
            return _ACCIDENTAL_NAMES[alter]
        return None

    return rule


def _note_xml(pitch, length, name, dots, divisions, voice, fermata, tie_in, tie_out,
              accidental=None) -> list[str]:
    staff, stem = _LAYOUT[voice]
    out = ["      <note>"]
    if pitch is None:
        out.append("        <rest/>")
    else:
        step, alter, octave = pitch
        out.append("        <pitch>")
        out.append(f"          <step>{step}</step>")
        if alter:
            out.append(f"          <alter>{alter}</alter>")
        out.append(f"          <octave>{octave}</octave>")
        out.append("        </pitch>")
    out.append(f"        <duration>{int(length * divisions)}</duration>")
    if tie_in:
        out.append('        <tie type="stop"/>')
    if tie_out:
        out.append('        <tie type="start"/>')
    out.append(f"        <voice>{voice + 1}</voice>")
    out.append(f"        <type>{name}</type>")
    out.extend(["        <dot/>"] * dots)
    if accidental is not None:
        out.append(f"        <accidental>{accidental}</accidental>")
    if pitch is not None:
        out.append(f"        <stem>{stem}</stem>")
    out.append(f"        <staff>{staff}</staff>")
    fermata = fermata and _FERMATA[voice] is not None
    if pitch is not None and (tie_in or tie_out or fermata):
        out.append("        <notations>")
        if tie_in:
            out.append('          <tied type="stop"/>')
        if tie_out:
            out.append('          <tied type="start"/>')
        if fermata:
            out.append(f'          <fermata type="{_FERMATA[voice]}"/>')
        out.append("        </notations>")
    out.append("      </note>")
    return out


def _voice_measures(tune, segs, voice, divisions, accidental_rule) -> list[list[str]]:
    """The note XML of one voice, split into bars."""
    bars = []
    si = 0
    carry = None  # the rest of a segment that crossed a barline
    for mi, measure in enumerate(tune.measures):
        lines: list[str] = []
        bar_end = measure.offset + measure.length
        at = measure.offset
        while at < bar_end:
            if carry is not None:
                seg, carry = carry, None
            elif si < len(segs):
                seg = list(segs[si])
                si += 1
            else:
                seg = [at, bar_end - at, None, False, False, False]
            off, length, pitch, fermata, tie_in, tie_out = seg
            if off > at:  # a gap in the source: fill with a rest
                carry = seg
                off, length, pitch, fermata, tie_in, tie_out = at, min(off, bar_end) - at, None, False, False, False
            here = min(length, bar_end - off)
            if here < length:
                carry = [off + here, length - here, pitch, False, pitch is not None, tie_out]
                tie_out = pitch is not None
            pieces = _pieces(here)
            for k, (value, name, dots) in enumerate(pieces):
                lines.extend(_note_xml(
                    pitch, value, name, dots, divisions, voice,
                    fermata and k == 0,
                    (tie_in if k == 0 else pitch is not None),
                    (tie_out if k == len(pieces) - 1 else pitch is not None),
                    None if pitch is None else accidental_rule(voice, mi, pitch),
                ))
            at = off + here
        bars.append(lines)
    return bars


def musicxml_string(chorale_id: int, chords) -> str:
    """The grand-staff MusicXML document for a chorale and its chosen chords."""
    tune = load_tune(chorale_id)
    chords = realise(chords)
    if len(chords) != len(tune.beats):
        raise ValueError(f"chorale {chorale_id} has {len(tune.beats)} beats, got {len(chords)} chords")
    parts = _segments(tune, chords)
    divisions = 1
    for segs in parts:
        for seg in segs:
            divisions = lcm(divisions, seg[0].denominator, seg[1].denominator)
    divisions = lcm(divisions, 16) if divisions > 8 else lcm(divisions, 8)
    rule = _accidental_rule(tune, parts)
    voices = [_voice_measures(tune, segs, v, divisions, rule) for v, segs in enumerate(parts)]

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<score-partwise version="3.1">',
        "  <work>",
        f"    <work-title>Chorale {chorale_id}</work-title>",
        "  </work>",
        "  <part-list>",
        f'    <score-part id="{PART_ID}">',
        "      <part-name>Chorale</part-name>",
        "    </score-part>",
        "  </part-list>",
        f'  <part id="{PART_ID}">',
    ]
    for mi, measure in enumerate(tune.measures):
        implicit = ' implicit="yes"' if measure.length != 4 else ""
        out.append(f'    <measure number="{measure.number}"{implicit}>')
        if mi == 0:
            out.extend([
                "      <attributes>",
                f"        <divisions>{divisions}</divisions>",
                "        <key>",
                f"          <fifths>{tune.key_signature}</fifths>",
                "        </key>",
                "        <time>",
                "          <beats>4</beats>",
                "          <beat-type>4</beat-type>",
                "        </time>",
                "        <staves>2</staves>",
                '        <clef number="1">',
                "          <sign>G</sign>",
                "          <line>2</line>",
                "        </clef>",
                '        <clef number="2">',
                "          <sign>F</sign>",
                "          <line>4</line>",
                "        </clef>",
                "      </attributes>",
            ])
        back = int(measure.length * divisions)
        for v in range(4):
            if v:
                out.extend(["      <backup>", f"        <duration>{back}</duration>", "      </backup>"])
            out.extend(voices[v][mi])
        out.append("    </measure>")
    out.append("  </part>")
    out.append("</score-partwise>")
    return "\n".join(out) + "\n"


def write_musicxml(chorale_id: int, chords, path) -> None:
    """Write the grand-staff MusicXML score for the chosen chords."""
    data = musicxml_string(chorale_id, chords).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)


def read_notes(path):
    """Every pitched note: (part id, staff, voice, onset, length, midi, step, alter).

    Onsets are in quarter lengths. Handles <backup>, <forward> and <chord>.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise DecodeError(f"cannot read MusicXML: {exc}") from exc
    notes = []
    for part in root.findall("part"):
        divisions = 1
        at = Fraction(0)
        last_onset = Fraction(0)
        for measure in part.findall("measure"):
            for el in measure:
                if el.tag == "attributes":
                    d = el.find("divisions")
                    if d is not None:
                        divisions = int(d.text)
                elif el.tag in ("backup", "forward"):
                    step = Fraction(int(el.find("duration").text), divisions)
                    at += step if el.tag == "forward" else -step
                elif el.tag == "note":
                    if el.find("grace") is not None or el.find("duration") is None:
                        continue
                    length = Fraction(int(el.find("duration").text), divisions)
                    is_chord = el.find("chord") is not None
                    onset = last_onset if is_chord else at
                    pitch = el.find("pitch")
                    if pitch is not None:
                        step = pitch.find("step").text
                        alter_el = pitch.find("alter")
                        alter = int(float(alter_el.text)) if alter_el is not None else 0
                        octave = int(pitch.find("octave").text)
                        midi = 12 * (octave + 1) + _STEP_PC[step] + alter
                        staff_el = el.find("staff")
                        voice_el = el.find("voice")
                        notes.append((
                            part.get("id"),
                            int(staff_el.text) if staff_el is not None else 1,
                            voice_el.text if voice_el is not None else "1",
                            onset, length, midi, step, alter,
                        ))
                    if not is_chord:
                        last_onset = at
                        at += length
    return notes


def _staves(notes):
    """Map the notes onto (upper staff, lower staff).

    A grand staff is one part with staves 1 and 2. A score with one part per
    voice (the older four-stave layout) maps parts 1-2 to the upper staff and
    parts 3-4 to the lower staff.
    """
    parts = sorted({n[0] for n in notes})
    if len(parts) == 1:
        return [n for n in notes if n[1] == 1], [n for n in notes if n[1] == 2]
    if len(parts) == 4:
        return [n for n in notes if n[0] in parts[:2]], [n for n in notes if n[0] in parts[2:]]
    raise DecodeError(f"expected a grand staff or four parts, found {len(parts)} parts")


def staff_pitches_at(notes, offset):
    """Distinct MIDI pitches sounding at an offset, highest first."""
    return sorted({n[5] for n in notes if n[3] <= offset < n[3] + n[4]}, reverse=True)


def score_end(path) -> Fraction:
    notes = read_notes(path)
    return max((n[3] + n[4] for n in notes), default=Fraction(0))


def chord_from_staves(upper, lower, where=""):
    """(S, A, T, B) from the pitches on each staff. One pitch means a shared note."""
    for name, pitches in (("treble", upper), ("bass", lower)):
        if not 1 <= len(pitches) <= 2:
            raise DecodeError(f"{len(pitches)} different pitches on the {name} staff{where}")
    s, a = upper[0], upper[-1]
    t, b = lower[0], lower[-1]
    return (s, a, t, b)


def read_musicxml(chorale_id: int, path) -> list[tuple[int, int, int, int]]:
    """The (S, A, T, B) chord written at every beat of the chorale."""
    tune = load_tune(chorale_id)
    upper, lower = _staves(read_notes(path))
    chords = []
    for beat in tune.beats:
        where = f" at phrase {beat.phrase_index}, beat {beat.beat_index} (offset {beat.offset})"
        chords.append(chord_from_staves(
            staff_pitches_at(upper, beat.offset), staff_pitches_at(lower, beat.offset), where
        ))
    return chords
