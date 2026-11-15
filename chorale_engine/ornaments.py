"""Experimental ornaments: quavers between beats in alto, tenor and bass.

    extra = ornaments(chorale_id, chords)   # per beat: (alto, tenor, bass) MIDI or None

An ornament sounds on the second half of a beat. The chord on the beat is
never changed, and the decoder only reads the notes on the beat, so a score
decodes the same with or without ornaments. The result is a pure function
of the chorale and the chords.

Kinds:
  - walking bass: a quaver in the bass on most beats, from a passing note,
    a chord tone, an arpeggio or an octave leap, so the bass keeps moving;
  - passing note: an inner voice that moves by a 3rd steps through the
    scale note in between;
  - neighbour note: an inner voice that repeats its note steps to the next
    scale note and back (only where the bass is not already walking).

Inner voices only move where the soprano holds its note (the voices take
turns, as in Bach). Nothing is added on a held cadence chord or next to a
rest. An ornament is dropped if it would cross another voice, sit a
semitone (or a major 7th) from another sounding voice, or make parallel
5ths or octaves into or out of the half beat.
"""

from __future__ import annotations

from fractions import Fraction

from .corpus import load_tune
from .keys import lof_to_pc, lof_to_step_alter

_STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def spell_in_key(key, midi: int) -> tuple[str, int, int]:
    """(step, alter, octave) of a pitch that belongs to the key."""
    for lof in key.degree_lofs():
        if lof_to_pc(lof) == midi % 12:
            step, alter = lof_to_step_alter(lof)
            return step, alter, (midi - alter - _STEP_PC[step]) // 12 - 1
    raise ValueError(f"pitch {midi} is not in {key}")


def _natural_pcs(key) -> frozenset[int]:
    """The plain scale of the key: major, or natural minor (no raised 6th or 7th)."""
    return frozenset(lof_to_pc(i) for i in key.degree_lofs()[:7])


def _scale_choice(key, candidates):
    """The first candidate in the plain scale, else the first in the key at all."""
    in_key = [m for m in candidates if m % 12 in key.pitch_classes()]
    plain = [m for m in in_key if m % 12 in _natural_pcs(key)]
    return plain[0] if plain else in_key[0] if in_key else None


def _passing(key, start: int, end: int):
    """The scale note between two pitches a 3rd apart, or None."""
    d = end - start
    if abs(d) not in (3, 4):
        return None
    sign = 1 if d > 0 else -1
    # Both steps must be a tone or a semitone.
    middles = [start + sign * 2] if abs(d) == 4 else [start + sign * 1, start + sign * 2]
    return _scale_choice(key, middles)


def _neighbours(key, pitch: int):
    """Lower then upper neighbour scale notes of a pitch."""
    out = []
    for sign in (-1, 1):
        m = _scale_choice(key, [pitch + sign * 2, pitch + sign * 1])
        if m is not None:
            out.append(m)
    return out


def _fits_all(off, cur, nxt, changed) -> bool:
    """Is the off-beat chord `off` (voices in `changed` moved from `cur`) good between `cur` and `nxt`?

    No crossing; no semitone or major 7th between a moved voice and any other; no parallel 5ths or octaves
    from the beat into the half beat, or from the half beat into the next beat, for a pair that includes
    a moved voice.
    """
    if not off[0] >= off[1] >= off[2] >= off[3]:
        return False
    for v in changed:
        for u in range(4):
            if u != v and (off[u] - off[v]) % 12 in (1, 11):
                return False
    for a_chord, b_chord in ((cur, off), (off, nxt)):
        for v in changed:
            for u in range(4):
                if u == v or b_chord[u] == a_chord[u] or b_chord[v] == a_chord[v]:
                    continue
                hi, lo = (u, v) if u < v else (v, u)
                before = (a_chord[hi] - a_chord[lo]) % 12
                after = (b_chord[hi] - b_chord[lo]) % 12
                if before == after and after in (0, 7):
                    return False
    return True


def _fits(voice: int, pitch: int, sounding, nxt) -> bool:
    """Can `voice` sing `pitch` on the off-beat, the others holding `sounding`, then all move to `nxt`?"""
    off = list(sounding)
    off[voice] = pitch
    return _fits_all(off, tuple(sounding), nxt, (voice,))


BASS_LOW, BASS_HIGH = 40, 60
#the biggest leap the walking bass takes to a chord tone inside a beat (5 = a perfect 4th)
MAX_BASS_LEAD = 5


def _bass_walk(key, cur, nxt, off):
    """The off-beat bass note that keeps the bass walking from cur[3] to nxt[3], or None.

    Best first: a passing note through a 3rd; a chord tone that then steps to the next bass;
    a chord tone a 3rd from the next bass (an arpeggio); the octave of the beat note when the
    next bass lies near it; a neighbour note when the bass repeats. Each must fit the voices
    sounding on the half beat (`off`, with the soprano's own half-beat note).
    """
    here, there = cur[3], nxt[3]
    chord_pcs = {m % 12 for m in cur}
    candidates = []  #(preference, cost, pitch)
    m = _passing(key, here, there)
    if m is not None:
        candidates.append((0, 0, m))
    for pitch in range(BASS_LOW, BASS_HIGH + 1):
        if pitch == here or pitch % 12 not in chord_pcs:
            continue
        lead, arrive = abs(pitch - here), abs(there - pitch)
        if lead > MAX_BASS_LEAD:
            continue
        if arrive in (1, 2):
            candidates.append((1, lead + arrive, pitch))
        elif arrive in (3, 4) and there != here:
            candidates.append((2, lead + arrive, pitch))
    for octave in (here - 12, here + 12):
        if BASS_LOW <= octave <= BASS_HIGH and abs(there - octave) <= 2:
            candidates.append((3, abs(there - octave), octave))
    if here == there:
        for n in _neighbours(key, here):
            candidates.append((4, abs(n - here), n))
    for _pref, _cost, pitch in sorted(candidates):
        if pitch % 12 not in key.pitch_classes():
            continue  # the opening or final home-tonic chord can hold notes outside the beat's key
        trial = list(off)
        trial[3] = pitch
        if _fits_all(trial, cur, nxt, (3,)):
            return pitch
    return None


def _soprano_at(tune, offset):
    for e in tune.soprano:
        if e.midi is not None and e.offset <= offset < e.offset + e.length:
            return e.midi
    return None


def ornaments(chorale_id: int, chords, walking: bool = True) -> list[tuple]:
    """Off-beat pitch for (alto, tenor, bass) at every beat, None where there is none.

    walking=True gives a walking bass: quavers in the bass wherever one fits, with passing
    notes, chord tones and octave leaps between the beats. At most one inner voice moves as well,
    and only where the soprano holds its note through the beat.
    """
    tune = load_tune(chorale_id)
    chords = [tuple(int(m) for m in c) for c in chords]
    if len(chords) != len(tune.beats):
        raise ValueError(f"chorale {chorale_id} has {len(tune.beats)} beats, got {len(chords)} chords")
    onsets = {e.offset for e in tune.soprano if e.midi is not None}
    out = [(None, None, None)] * len(chords)
    last_neighbour = {1: -10, 2: -10}
    for i in range(len(chords) - 1):
        beat, nxt_beat = tune.beats[i], tune.beats[i + 1]
        if beat.is_cadence or nxt_beat.offset != beat.offset + 1:
            continue
        cur, nxt = chords[i], chords[i + 1]
        soprano_moves = any(beat.offset < o < beat.offset + 1 for o in onsets)
        half = _soprano_at(tune, beat.offset + Fraction(1, 2))
        off = list(cur)
        if half is not None:
            off[0] = half
        row = [None, None, None]
        if walking:
            m = _bass_walk(beat.key, cur, nxt, off)
            if m is not None:
                off[3] = m
                row[2] = m
        if soprano_moves:
            # The voices take turns: with the soprano moving, only the bass may move too.
            out[i] = tuple(row)
            continue
        chosen = None
        order = (2, 1) if walking else (3, 2, 1)
        for v in order:
            if v == 3 and row[2] is not None:
                continue
            m = _passing(beat.key, cur[v], nxt[v])
            if m is not None:
                trial = list(off)
                trial[v] = m
                moved = (v,) + ((3,) if row[2] is not None else ())
                if _fits_all(trial, cur, nxt, moved):
                    chosen = (v, m)
                    break
        if chosen is None and (not walking or row[2] is None):
            for v in (1, 2):  # alto, tenor; not two beats running in one voice
                if cur[v] != nxt[v] or i - last_neighbour[v] < 2:
                    continue
                for m in _neighbours(beat.key, cur[v]):
                    trial = list(off)
                    trial[v] = m
                    if _fits_all(trial, cur, nxt, (v,)):
                        chosen = (v, m)
                        last_neighbour[v] = i
                        break
                if chosen is not None:
                    break
        if chosen is not None:
            row[chosen[0] - 1] = chosen[1]
        out[i] = tuple(row)
    return out


# ---------------------------------------------------------------------------
# Ornament signs on the soprano: trill, turn, mordent, inverted mordent.
# These are printed over a note and played by the performer; the written note
# is unchanged, so they never affect decoding.
# ---------------------------------------------------------------------------

#MusicXML element names, best first: a note gets at most one sign
TRILL, TURN, MORDENT, INVERTED_MORDENT = "trill-mark", "turn", "mordent", "inverted-mordent"


def _soprano_onsets(tune):
    """Written soprano note index at each beat where a new (untied) note starts."""
    starts = {}
    for i, e in enumerate(tune.soprano):
        if e.midi is not None and e.tie in (None, "start") and e.offset.denominator == 1:
            starts[int(e.offset)] = i
    return starts


def _previous_note(tune, index):
    for j in range(index - 1, -1, -1):
        e = tune.soprano[j]
        if e.midi is None:
            return None  # a rest breaks the line
        if e.tie in (None, "start"):
            return e
    return None


def _sounding_length(tune, index):
    """Length of a soprano note including the notes tied on to it."""
    total = tune.soprano[index].length
    j = index
    while tune.soprano[j].tie in ("start", "continue") and j + 1 < len(tune.soprano):
        j += 1
        total += tune.soprano[j].length
    return total


def ornament_signs(chorale_id: int) -> dict:
    """{soprano note index: MusicXML ornament name}. Depends only on the tune, so it is the same for every number."""
    tune = load_tune(chorale_id)
    starts = _soprano_onsets(tune)
    signs = {}
    # Cadential trills: the note before each fermata, falling a step to it or rising from the leading tone.
    for p in range(tune.n_phrases):
        beats = tune.phrase_beats(p)
        if len(beats) < 2:
            continue
        before, cadence = beats[-2], beats[-1]
        index = starts.get(before.offset)
        if index is None or _sounding_length(tune, index) < 1:
            continue
        step = cadence.soprano_midi - tune.soprano[index].midi
        if step in (-1, -2) or step == 1:
            signs[index] = TRILL
    # One turn per piece: a long note (two beats or more) that moves on by step.
    for beat in tune.beats:
        index = starts.get(beat.offset)
        if index is None or index in signs or beat.is_cadence:
            continue
        nxt = next((e for e in tune.soprano[index + 1:] if e.midi is not None and e.tie in (None, "start")), None)
        if _sounding_length(tune, index) >= 2 and nxt is not None and abs(nxt.midi - tune.soprano[index].midi) in (1, 2):
            signs[index] = TURN
            break
    # At most one mordent and one inverted mordent per phrase, away from the cadence.
    for p in range(tune.n_phrases):
        beats = tune.phrase_beats(p)
        placed = set()
        for beat in beats[1:-2]:
            index = starts.get(beat.offset)
            if index is None or index in signs or _sounding_length(tune, index) < 1:
                continue
            prev = _previous_note(tune, index)
            if prev is None:
                continue
            move = tune.soprano[index].midi - prev.midi
            if MORDENT not in placed and move >= 3:
                signs[index] = MORDENT
                placed.add(MORDENT)
            elif INVERTED_MORDENT not in placed and move in (-1, -2) and _sounding_length(tune, index) == 1:
                signs[index] = INVERTED_MORDENT
                placed.add(INVERTED_MORDENT)
    return dict(sorted(signs.items()))


def neighbour_pitches(key, midi: int) -> tuple[int, int]:
    """(upper, lower) scale neighbours of a pitch, for playing an ornament."""
    upper = _scale_choice(key, [midi + 2, midi + 1]) or midi + 2
    lower = _scale_choice(key, [midi - 2, midi - 1]) or midi - 2
    return upper, lower


def play_sign(sign: str, key, midi: int, length: Fraction) -> list[tuple[Fraction, int]]:
    """The notes a performer plays for one ornamented note: [(duration, MIDI)], adding up to length."""
    upper, lower = neighbour_pitches(key, midi)
    short = Fraction(1, 8)  # a demisemiquaver
    if sign == TRILL:
        #from the upper note, as in the baroque, for three quarters of the note, then the main note
        notes, used, flip = [], Fraction(0), True
        while used + short <= length * Fraction(3, 4):
            notes.append((short, upper if flip else midi))
            used += short
            flip = not flip
        return notes + [(length - used, midi)]
    if sign == MORDENT:
        return [(short, midi), (short, lower), (length - 2 * short, midi)]
    if sign == INVERTED_MORDENT:
        return [(short, midi), (short, upper), (length - 2 * short, midi)]
    if sign == TURN:
        q = Fraction(1, 4)
        return [(q, upper), (q, midi), (q, lower), (length - 3 * q, midi)]
    return [(length, midi)]
