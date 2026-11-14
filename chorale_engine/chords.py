"""Chord vocabulary and voicing generation (incremental and brute force).

A chord is a tuple (S, A, T, B) of MIDI ints. A voicing is a chord that
belongs to the vocabulary of its beat: every note is a chord tone, the
required tones are present, the inversion is allowed and every voice is in
range. The hard rules 1-11 are applied separately, in rules.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .keys import Key, lof_to_pc, lof_to_step_alter

A_RANGE = (55, 74)  # G3-D5
T_RANGE = (48, 67)  # C3-G4
B_RANGE = (40, 60)  # E2-C4
MAX_INNER_LEAP = 5  # alto and tenor move at most a perfect 4th (rule 8)

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def canonical_key(chord) -> tuple[int, int, int, int]:
    """Canonical sort key of a chord: (B, T, A, S)."""
    return (chord[3], chord[2], chord[1], chord[0])


@dataclass(frozen=True)
class BeatContext:
    """Everything the vocabulary and the rules need to know about one beat."""

    key: Key
    soprano_midi: int
    home_key: Key
    is_opening: bool = False  # first beat of the piece: the home tonic triad is also allowed
    is_final: bool = False  # final cadence beat: home tonic and Picardy third also allowed


@dataclass(frozen=True)
class ChordType:
    """A chord of the vocabulary, as line-of-fifths indices (root, 3rd, 5th[, 7th])."""

    name: str
    lofs: tuple[int, ...]
    degree: int  # root scale degree in the beat's key, 0 for a chord from outside it
    dominant: bool = False  # V, V7, vii°, viiø7, vii°7

    @property
    def pcs(self) -> tuple[int, ...]:
        return tuple(lof_to_pc(i) for i in self.lofs)

    @property
    def is_seventh(self) -> bool:
        return len(self.lofs) == 4

    @property
    def root_pc(self) -> int:
        return lof_to_pc(self.lofs[0])

    @property
    def third_pc(self) -> int:
        return lof_to_pc(self.lofs[1])

    @property
    def seventh_pc(self) -> int | None:
        return lof_to_pc(self.lofs[3]) if self.is_seventh else None

    @property
    def quality(self) -> str:
        """Triad quality of root, 3rd and 5th: 'major', 'minor' or 'diminished'."""
        third = (self.pcs[1] - self.pcs[0]) % 12
        fifth = (self.pcs[2] - self.pcs[0]) % 12
        if (third, fifth) == (4, 7):
            return "major"
        if (third, fifth) == (3, 7):
            return "minor"
        if (third, fifth) == (3, 6):
            return "diminished"
        raise ValueError(f"unsupported chord {self.name}")

    @property
    def inversions(self) -> tuple[int, ...]:
        """Chord-tone indices allowed in the bass (0 = root position)."""
        if self.is_seventh:
            return (0, 1, 2, 3)
        if self.quality == "diminished":
            return (1,)  # diminished triads only in first inversion
        return (0, 1, 2)

    @property
    def required_pcs(self) -> tuple[int, ...]:
        """Root and 3rd, plus the 7th of a seventh chord. The 5th may be omitted."""
        p = self.pcs
        return (p[0], p[1], p[3]) if self.is_seventh else (p[0], p[1])


def home_tonic_type(home: Key, picardy: bool = False) -> ChordType:
    h = home.lof
    if home.mode == "major":
        return ChordType("home I", (h, h + 4, h + 1), 0)
    if picardy:
        return ChordType("home I (Picardy)", (h, h + 4, h + 1), 0)
    return ChordType("home i", (h, h - 3, h + 1), 0)


@lru_cache(maxsize=None)
def vocabulary(ctx: BeatContext) -> tuple[ChordType, ...]:
    """Chord types allowed at a beat, in a fixed order."""
    t = ctx.key.lof
    if ctx.key.mode == "major":
        d = [None] + [t + x for x in (0, 2, 4, -1, 1, 3, 5)]  # d[1]..d[7]
        types = [
            ChordType("I", (d[1], d[3], d[5]), 1),
            ChordType("ii", (d[2], d[4], d[6]), 2),
            ChordType("iii", (d[3], d[5], d[7]), 3),
            ChordType("IV", (d[4], d[6], d[1]), 4),
            ChordType("V", (d[5], d[7], d[2]), 5, True),
            ChordType("vi", (d[6], d[1], d[3]), 6),
            ChordType("vii°", (d[7], d[2], d[4]), 7, True),
            ChordType("V7", (d[5], d[7], d[2], d[4]), 5, True),
            ChordType("ii7", (d[2], d[4], d[6], d[1]), 2),
            ChordType("viiø7", (d[7], d[2], d[4], d[6]), 7, True),
        ]
    else:
        n = [None] + [t + x for x in (0, 2, -3, -1, 1, -4, -2)]  # natural minor n[1]..n[7]
        r6, r7 = t + 3, t + 5
        types = [
            ChordType("i", (n[1], n[3], n[5]), 1),
            ChordType("ii°", (n[2], n[4], n[6]), 2),
            ChordType("III", (n[3], n[5], n[7]), 3),
            ChordType("iv", (n[4], n[6], n[1]), 4),
            ChordType("V", (n[5], r7, n[2]), 5, True),
            ChordType("VI", (n[6], n[1], n[3]), 6),
            ChordType("vii°", (r7, n[2], n[4]), 7, True),
            ChordType("V7", (n[5], r7, n[2], n[4]), 5, True),
            ChordType("iiø7", (n[2], n[4], n[6], n[1]), 2),
            ChordType("vii°7", (r7, n[2], n[4], n[6]), 7, True),
        ]
        if ctx.soprano_midi % 12 == lof_to_pc(r6):
            # Only under a raised 6th in the soprano.
            types.append(ChordType("IV", (n[4], r6, n[1]), 4))
            types.append(ChordType("ii", (n[2], n[4], r6), 2))

    extras = []
    if ctx.is_opening or ctx.is_final:
        extras.append(home_tonic_type(ctx.home_key))
    if ctx.is_final and ctx.home_key.mode == "minor":
        extras.append(home_tonic_type(ctx.home_key, picardy=True))
    for extra in extras:
        if not any(frozenset(x.pcs) == frozenset(extra.pcs) for x in types):
            types.append(extra)
    return tuple(types)


def is_home_tonic(ctx: BeatContext, ctype: ChordType, picardy: bool | None = None) -> bool:
    """Is this chord type the home tonic triad?

    picardy=None accepts either form, False only the plain tonic, True only
    the major tonic of a minor home key.
    """
    if ctype.is_seventh:
        return False
    pcs = frozenset(ctype.pcs)
    plain = frozenset(home_tonic_type(ctx.home_key).pcs)
    if ctx.home_key.mode == "major":
        return pcs == plain and picardy is not True
    major = frozenset(home_tonic_type(ctx.home_key, picardy=True).pcs)
    if picardy is None:
        return pcs in (plain, major)
    return pcs == (major if picardy else plain)


@dataclass(frozen=True)
class Analysis:
    """A voicing identified against the vocabulary of its beat."""

    chord: tuple[int, int, int, int]
    ctype: ChordType
    inversion: int  # chord-tone index in the bass
    lofs: tuple[int, int, int, int]  # line-of-fifths index of S, A, T, B (the spelling)

    @property
    def is_second_inversion_triad(self) -> bool:
        return self.inversion == 2 and not self.ctype.is_seventh


def _in_range(chord) -> bool:
    return (
        A_RANGE[0] <= chord[1] <= A_RANGE[1]
        and T_RANGE[0] <= chord[2] <= T_RANGE[1]
        and B_RANGE[0] <= chord[3] <= B_RANGE[1]
    )


@lru_cache(maxsize=None)
def _analyse(ctx: BeatContext, chord: tuple[int, int, int, int]) -> Analysis | None:
    if chord[0] != ctx.soprano_midi or not _in_range(chord):
        return None
    pcs = tuple(m % 12 for m in chord)
    present = set(pcs)
    for ctype in vocabulary(ctx):
        tones = ctype.pcs
        if not present.issubset(tones):
            continue
        if any(r not in present for r in ctype.required_pcs):
            continue
        inversion = tones.index(pcs[3])
        if inversion not in ctype.inversions:
            continue
        lofs = tuple(ctype.lofs[tones.index(pc)] for pc in pcs)
        return Analysis(chord, ctype, inversion, lofs)
    return None


def analyse(ctx: BeatContext, chord) -> Analysis | None:
    """Identify a chord at a beat, or None if it is not a voicing there."""
    return _analyse(ctx, tuple(int(m) for m in chord))


def spell(ctx: BeatContext, chord) -> tuple[tuple[str, int, int], ...]:
    """Deterministic spelling of a voicing: (step, alter, octave) for S, A, T, B."""
    info = analyse(ctx, chord)
    if info is None:
        raise ValueError(f"{tuple(chord)} is not a voicing at this beat")
    out = []
    for midi, lof in zip(info.chord, info.lofs):
        step, alter = lof_to_step_alter(lof)
        out.append((step, alter, (midi - alter - _LETTER_PC[step]) // 12 - 1))
    return tuple(out)


def _voicings(ctx: BeatContext, a_range, t_range) -> list[tuple[int, int, int, int]]:
    """Every voicing with A and T inside the given windows, in canonical order."""
    s = ctx.soprano_midi
    spc = s % 12
    found = set()
    for ctype in vocabulary(ctx):
        tones = ctype.pcs
        if spc not in tones:
            continue
        required = ctype.required_pcs
        bass_pcs = {tones[i] for i in ctype.inversions}
        altos = [m for m in range(a_range[0], a_range[1] + 1) if m % 12 in tones]
        tenors = [m for m in range(t_range[0], t_range[1] + 1) if m % 12 in tones]
        basses = [m for m in range(B_RANGE[0], B_RANGE[1] + 1) if m % 12 in bass_pcs]
        for a in altos:
            for t in tenors:
                for b in basses:
                    present = (spc, a % 12, t % 12, b % 12)
                    if all(r in present for r in required):
                        found.add((s, a, t, b))
    return sorted(found, key=canonical_key)


@lru_cache(maxsize=None)
def brute_force_voicings(ctx: BeatContext) -> tuple[tuple[int, int, int, int], ...]:
    """Every voicing at a beat, with no hard rule applied, in canonical order."""
    return tuple(_voicings(ctx, A_RANGE, T_RANGE))


def incremental_voicings(ctx: BeatContext, prev_chord, max_leap: int | None = MAX_INNER_LEAP):
    """Voicings reachable from the previous chord.

    S is fixed by the tune, A and T stay within max_leap semitones of their
    previous pitch, B is anywhere in its range. With no previous chord (or
    max_leap=None) this is every voicing.
    """
    if prev_chord is None or max_leap is None:
        return list(brute_force_voicings(ctx))
    a_range = (max(A_RANGE[0], prev_chord[1] - max_leap), min(A_RANGE[1], prev_chord[1] + max_leap))
    t_range = (max(T_RANGE[0], prev_chord[2] - max_leap), min(T_RANGE[1], prev_chord[2] + max_leap))
    return _voicings(ctx, a_range, t_range)


def beat_context(tune, index: int) -> BeatContext:
    """BeatContext of beat `index` (position in the whole piece) of a Tune."""
    beat = tune.beats[index]
    return BeatContext(
        key=beat.key,
        soprano_midi=beat.soprano_midi,
        home_key=tune.home_key,
        is_opening=index == 0,
        is_final=index == len(tune.beats) - 1,
    )
