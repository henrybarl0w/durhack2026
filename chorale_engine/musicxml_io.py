"""MusicXML export and parse back to per-beat (S, A, T, B) tuples.

The file is written by hand so that it is byte-identical for the same
input: four parts S, A, T, B, one staff and one voice each, explicit part
IDs, a fixed title, no dates and no generated IDs.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction
from math import lcm

from .chords import beat_context, spell
from .corpus import load_tune
from .errors import DecodeError
from .realise import realise

PART_IDS = ("P1", "P2", "P3", "P4")
PART_NAMES = ("S", "A", "T", "B")
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


def _note_xml(pitch, length, name, dots, divisions, fermata, tie_in, tie_out) -> list[str]:
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
    out.append("        <voice>1</voice>")
    out.append(f"        <type>{name}</type>")
    out.extend(["        <dot/>"] * dots)
    if pitch is not None and (tie_in or tie_out or fermata):
        out.append("        <notations>")
        if tie_in:
            out.append('          <tied type="stop"/>')
        if tie_out:
            out.append('          <tied type="start"/>')
        if fermata:
            out.append('          <fermata type="upright"/>')
        out.append("        </notations>")
    out.append("      </note>")
    return out


def musicxml_string(chorale_id: int, chords) -> str:
    """The MusicXML document for a chorale and its chosen chords."""
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

    clefs = [("G", 2, 0), ("G", 2, 0), ("G", 2, -1), ("F", 4, 0)]
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<score-partwise version="3.1">',
        "  <work>",
        f"    <work-title>Chorale {chorale_id}</work-title>",
        "  </work>",
        "  <part-list>",
    ]
    for pid, name in zip(PART_IDS, PART_NAMES):
        out.append(f'    <score-part id="{pid}">')
        out.append(f"      <part-name>{name}</part-name>")
        out.append("    </score-part>")
    out.append("  </part-list>")

    for pid, segs, (sign, line, octave_change) in zip(PART_IDS, parts, clefs):
        out.append(f'  <part id="{pid}">')
        si = 0
        carry = None  # the rest of a segment that crossed a barline
        for mi, measure in enumerate(tune.measures):
            implicit = ' implicit="yes"' if measure.length != 4 else ""
            out.append(f'    <measure number="{measure.number}"{implicit}>')
            if mi == 0:
                out.append("      <attributes>")
                out.append(f"        <divisions>{divisions}</divisions>")
                out.append("        <key>")
                out.append(f"          <fifths>{tune.key_signature}</fifths>")
                out.append("        </key>")
                out.append("        <time>")
                out.append("          <beats>4</beats>")
                out.append("          <beat-type>4</beat-type>")
                out.append("        </time>")
                out.append("        <staves>1</staves>")
                out.append("        <clef>")
                out.append(f"          <sign>{sign}</sign>")
                out.append(f"          <line>{line}</line>")
                if octave_change:
                    out.append(f"          <clef-octave-change>{octave_change}</clef-octave-change>")
                out.append("        </clef>")
                out.append("      </attributes>")
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
                    out.extend(_note_xml(
                        pitch, value, name, dots, divisions,
                        fermata and k == 0,
                        (tie_in if k == 0 else pitch is not None),
                        (tie_out if k == len(pieces) - 1 else pitch is not None),
                    ))
                at = off + here
            out.append("    </measure>")
        out.append("  </part>")
    out.append("</score-partwise>")
    return "\n".join(out) + "\n"


def write_musicxml(chorale_id: int, chords, path) -> None:
    """Write the four-stave MusicXML score for the chosen chords."""
    data = musicxml_string(chorale_id, chords).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(data)


def read_parts(path):
    """Notes of every part: {part id: [(offset, length, midi, step, alter)]} in quarter lengths."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise DecodeError(f"cannot parse MusicXML: {exc}") from exc
    parts = {}
    for part in root.findall("part"):
        notes = []
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
                    if el.find("grace") is not None:
                        continue
                    length = Fraction(int(el.find("duration").text), divisions)
                    onset = last_onset if el.find("chord") is not None else at
                    pitch = el.find("pitch")
                    if pitch is not None:
                        step = pitch.find("step").text
                        alter_el = pitch.find("alter")
                        alter = int(float(alter_el.text)) if alter_el is not None else 0
                        octave = int(pitch.find("octave").text)
                        midi = 12 * (octave + 1) + _STEP_PC[step] + alter
                        notes.append((onset, length, midi, step, alter))
                    if el.find("chord") is None:
                        last_onset = at
                        at += length
        parts[part.get("id")] = notes
    return parts


def read_musicxml(chorale_id: int, path) -> list[tuple[int, int, int, int]]:
    """The (S, A, T, B) chord written at every beat of the chorale."""
    tune = load_tune(chorale_id)
    parts = read_parts(path)
    missing = [pid for pid in PART_IDS if pid not in parts]
    if missing:
        raise DecodeError(f"MusicXML has no part {', '.join(missing)}")
    chords = []
    for beat in tune.beats:
        chord = []
        for pid, name in zip(PART_IDS, PART_NAMES):
            sounding = [n for n in parts[pid] if n[0] <= beat.offset < n[0] + n[1]]
            if len(sounding) != 1:
                raise DecodeError(
                    f"part {name} has {len(sounding)} notes at phrase {beat.phrase_index}, "
                    f"beat {beat.beat_index} (offset {beat.offset})"
                )
            chord.append(sounding[0][2])
        chords.append(tuple(chord))
    return chords
