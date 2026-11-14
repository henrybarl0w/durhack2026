"""Independent checker for rules 1-11, run on a MusicXML file.

Written from the spec alone. It does not import rules.py, chords.py or
options.py. It reads the beats and keys of the tune from the corpus, and the
written notes (with their spellings, for rule 11) from the MusicXML.

    problems = check_musicxml(chorale_id, path)   # [] when the score is clean
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from fractions import Fraction

from .corpus import load_tune

_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_RANGES = {1: (55, 74), 2: (48, 67), 3: (40, 60)}
_NAMES = "SATB"

# Chords as semitones above the tonic: (name, tones, is dominant function).
_MAJOR = [
    ("I", (0, 4, 7), False), ("ii", (2, 5, 9), False), ("iii", (4, 7, 11), False),
    ("IV", (5, 9, 0), False), ("V", (7, 11, 2), True), ("vi", (9, 0, 4), False),
    ("vii°", (11, 2, 5), True), ("V7", (7, 11, 2, 5), True), ("ii7", (2, 5, 9, 0), False),
    ("viiø7", (11, 2, 5, 9), True),
]
_MINOR = [
    ("i", (0, 3, 7), False), ("ii°", (2, 5, 8), False), ("III", (3, 7, 10), False),
    ("iv", (5, 8, 0), False), ("V", (7, 11, 2), True), ("VI", (8, 0, 3), False),
    ("vii°", (11, 2, 5), True), ("V7", (7, 11, 2, 5), True), ("iiø7", (2, 5, 8, 0), False),
    ("vii°7", (11, 2, 5, 8), True),
]
_MINOR_RAISED_6 = [("IV", (5, 9, 0), False), ("ii", (2, 5, 9), False)]


def _read(path):
    """{voice: [(onset, length, midi, step, alter)]} for voices S, A, T, B (keys 0-3).

    A grand staff is one part with voices 1-4. The older layout is four parts.
    """
    root = ET.parse(path).getroot()
    parts = root.findall("part")
    voices = {0: [], 1: [], 2: [], 3: []}
    for pi, part in enumerate(parts):
        divisions, at = 1, Fraction(0)
        for measure in part.findall("measure"):
            for el in measure:
                if el.tag == "attributes" and el.find("divisions") is not None:
                    divisions = int(el.find("divisions").text)
                elif el.tag in ("backup", "forward"):
                    d = Fraction(int(el.find("duration").text), divisions)
                    at += d if el.tag == "forward" else -d
                elif el.tag == "note" and el.find("grace") is None:
                    length = Fraction(int(el.find("duration").text), divisions)
                    pitch = el.find("pitch")
                    if pitch is not None:
                        step = pitch.find("step").text
                        alter = pitch.find("alter")
                        alter = int(float(alter.text)) if alter is not None else 0
                        midi = 12 * (int(pitch.find("octave").text) + 1) + _STEP_PC[step] + alter
                        if len(parts) == 1:
                            v = int(el.find("voice").text) - 1
                        else:
                            v = pi
                        if v in voices:
                            voices[v].append((at, length, midi, step, alter))
                    if el.find("chord") is None:
                        at += length
    return voices


def _identify(key, home, soprano_pc, chord, first, final):
    """(tones as pitch classes, dominant?, tonic of the beat's key?, seventh?) or None."""
    t = key.tonic_pc
    table = list(_MAJOR if key.mode == "major" else _MINOR)
    if key.mode == "minor" and soprano_pc == (t + 9) % 12:
        table += _MINOR_RAISED_6
    cands = [(tuple((t + x) % 12 for x in tones), dom, name in ("I", "i")) for name, tones, dom in table]
    h = home.tonic_pc
    if first or final:
        third = 4 if home.mode == "major" else 3
        cands.append(((h, (h + third) % 12, (h + 7) % 12), False, False))
    if final and home.mode == "minor":
        cands.append(((h, (h + 4) % 12, (h + 7) % 12), False, False))
    pcs = [m % 12 for m in chord]
    for tones, dom, tonic in cands:
        if any(pc not in tones for pc in pcs):
            continue
        need = (tones[0], tones[1]) + ((tones[3],) if len(tones) == 4 else ())
        if any(r not in pcs for r in need):
            continue
        diminished = len(tones) == 3 and (tones[1] - tones[0]) % 12 == 3 and (tones[2] - tones[0]) % 12 == 6
        if diminished and pcs[3] != tones[1]:
            continue  # diminished triads only in first inversion
        return tones, dom, tonic, len(tones) == 4
    return None


def _step(a, b):
    d = b - a
    return 1 if d in (1, 2) else -1 if d in (-1, -2) else 0


def check_musicxml(chorale_id: int, path) -> list[str]:
    """Every problem found in the score, as text. An empty list means it passes."""
    tune = load_tune(chorale_id)
    parts = _read(path)
    problems = []
    ids = (0, 1, 2, 3)
    for pid in ids:
        if not parts[pid]:
            return [f"voice {_NAMES[pid]} is missing"]

    # Collect the chord at every beat, with spellings.
    rows = []
    for beat in tune.beats:
        notes = []
        for pid in ids:
            sounding = [n for n in parts[pid] if n[0] <= beat.offset < n[0] + n[1]]
            if len(sounding) != 1:
                return [f"beat {beat.index}: voice {_NAMES[pid]} has {len(sounding)} notes sounding"]
            notes.append(sounding[0])
        rows.append(notes)

    home = tune.home_key
    last = len(tune.beats) - 1
    infos = []
    for beat, notes in zip(tune.beats, rows):
        where = f"phrase {beat.phrase_index} beat {beat.beat_index}"
        chord = tuple(n[2] for n in notes)
        if chord[0] != beat.soprano_midi:
            problems.append(f"{where}: soprano is {chord[0]}, the tune has {beat.soprano_midi}")
        for v, (lo, hi) in _RANGES.items():
            if not lo <= chord[v] <= hi:
                problems.append(f"{where}: {_NAMES[v]} out of range")
        info = _identify(beat.key, home, chord[0] % 12, chord, beat.index == 0, beat.index == last)
        infos.append(info)
        if info is None:
            problems.append(f"{where}: {chord} is not a chord of the vocabulary")
            continue
        tones, _dom, _tonic, seventh = info
        s, a, t, b = chord
        if s - a > 12 or a - t > 12:
            problems.append(f"{where}: rule 3 spacing")
        if not s >= a >= t >= b:
            problems.append(f"{where}: rule 4 voice crossing")
        lt = (beat.key.tonic_pc + 11) % 12
        if sum(1 for m in chord if m % 12 == lt) > 1:
            problems.append(f"{where}: rule 5 doubled leading tone")
        major = not seventh and (tones[1] - tones[0]) % 12 == 4 and (tones[2] - tones[0]) % 12 == 7
        if major and sum(1 for m in chord if m % 12 == tones[1]) > 1:
            problems.append(f"{where}: rule 7 doubled 3rd in a major triad")
        if beat.is_cadence and (seventh or b % 12 != tones[0]):
            problems.append(f"{where}: cadence chord is not a root-position triad")
        if beat.index == last:
            third = {(home.tonic_pc + 4) % 12} | ({(home.tonic_pc + 3) % 12} if home.mode == "minor" else set())
            if seventh or tones[0] != home.tonic_pc or tones[1] not in third or (tones[2] - tones[0]) % 12 != 7:
                problems.append(f"{where}: final chord is not the home tonic")

    def six_four(i):
        info = infos[i]
        return info is not None and not info[3] and rows[i][3][2] % 12 == info[0][2]

    for i in range(len(rows)):
        beat = tune.beats[i]
        where = f"phrase {beat.phrase_index} beat {beat.beat_index}"
        cur = tuple(n[2] for n in rows[i])
        if infos[i] is None:
            continue
        if i == 0:
            if six_four(0) and not infos[0][2]:
                problems.append(f"{where}: rule 9 second inversion at the start")
            continue
        if infos[i - 1] is None:
            continue
        prev = tuple(n[2] for n in rows[i - 1])
        pkey = tune.beats[i - 1].key
        move = [c - p for p, c in zip(prev, cur)]

        for x in range(4):
            for y in range(x + 1, 4):
                if move[x] and move[y]:
                    before, after = (prev[x] - prev[y]) % 12, (cur[x] - cur[y]) % 12
                    if before == after and after in (0, 7):
                        problems.append(f"{where}: rule 1 parallel {_NAMES[x]}/{_NAMES[y]}")
        if move[0] * move[3] > 0 and abs(move[0]) > 2 and (cur[0] - cur[3]) % 12 in (0, 7):
            problems.append(f"{where}: rule 2 hidden 5th or octave")
        for hi, lo in ((0, 1), (1, 2), (2, 3)):
            if cur[lo] > prev[hi] or cur[hi] < prev[lo]:
                problems.append(f"{where}: rule 4 overlap {_NAMES[hi]}/{_NAMES[lo]}")
        ptones, pdom, ptonic, pseventh = infos[i - 1]
        if pdom and prev[3] % 12 == (pkey.tonic_pc + 11) % 12 and move[3] != 1:
            problems.append(f"{where}: rule 5 leading tone in the bass does not rise")
        if pseventh:
            for v in (1, 2, 3):
                if prev[v] % 12 == ptones[3] and move[v] not in (-1, -2):
                    problems.append(f"{where}: rule 6 the 7th in {_NAMES[v]} does not resolve")
        for v in (1, 2):
            if abs(move[v]) > 5:
                problems.append(f"{where}: rule 8 leap in {_NAMES[v]}")
        bass_step = _step(prev[3], cur[3])
        if six_four(i) and not infos[i][2] and bass_step == 0:
            problems.append(f"{where}: rule 9 second inversion approached by leap")
        if six_four(i - 1):
            pt = pkey.tonic_pc
            v_pcs = {(pt + 7) % 12, (pt + 11) % 12, (pt + 2) % 12}
            cadential = ptonic and cur[3] == prev[3] and set(infos[i][0]) in (v_pcs, v_pcs | {(pt + 5) % 12})
            passing = i >= 2 and bass_step != 0 and _step(rows[i - 2][3][2], prev[3]) == bass_step
            if not (cadential or passing):
                problems.append(f"{where}: rule 9 second inversion neither cadential nor passing")
        if beat.key.mode == "minor":
            pair = {(beat.key.tonic_pc + 8) % 12, (beat.key.tonic_pc + 11) % 12}
            for v in (1, 2, 3):
                if abs(move[v]) == 3 and {prev[v] % 12, cur[v] % 12} == pair:
                    problems.append(f"{where}: rule 10 augmented 2nd in {_NAMES[v]}")
        # Rule 11 from the written spellings.
        for x in range(4):
            _o, _l, _m, pstep, palter = rows[i - 1][x]
            for y in range(4):
                if y == x:
                    continue
                cstep, calter = rows[i][y][3], rows[i][y][4]
                if cstep != pstep or calter == palter:
                    continue
                own = rows[i][x]
                chromatic = own[3] == cstep and own[4] == calter and abs(move[x]) == abs(calter - palter)
                if not chromatic:
                    problems.append(f"{where}: rule 11 cross-relation {_NAMES[x]}/{_NAMES[y]}")
    return problems
