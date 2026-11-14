"""Hard rules 1-11.

Every check returns the set of rule numbers a chord violates, so the survey
can blame single rules and switch them off one at a time. A chord is legal
when the set is empty (after removing disabled rules).

Rules about one voice's motion (5, 6, 8, 10) apply to A, T and B only.
Rules about pairs of voices (1, 2, 3, 4, 11) include the soprano.
"""

from __future__ import annotations

from functools import lru_cache

from .chords import (
    MAX_INNER_LEAP,
    Analysis,
    BeatContext,
    analyse,
    brute_force_voicings,
    canonical_key,
    incremental_voicings,
)
from .keys import lof_to_pc

ALL_RULES = tuple(range(1, 12))
NO_RULES: frozenset[int] = frozenset()

_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
_LOWER = (1, 2, 3)  # A, T, B


def leading_tone_pc(key) -> int:
    """Leading tone of a key (the raised 7th in minor)."""
    return (key.tonic_pc + 11) % 12


def _is_step(a: int, b: int) -> int:
    """+1 for a step up from a to b, -1 for a step down, 0 otherwise."""
    d = b - a
    if d in (1, 2):
        return 1
    if d in (-1, -2):
        return -1
    return 0


def _need(ctx: BeatContext, chord) -> Analysis:
    info = analyse(ctx, chord)
    if info is None:
        raise ValueError(f"{tuple(chord)} is not a voicing at this beat")
    return info


@lru_cache(maxsize=None)
def _static(ctx: BeatContext, chord) -> frozenset[int]:
    info = _need(ctx, chord)
    s, a, t, b = chord
    bad = set()
    # 3. Spacing.
    if s - a > 12 or a - t > 12:
        bad.add(3)
    # 4. No voice crossing.
    if not (s >= a >= t >= b):
        bad.add(4)
    # 5. Leading tone never doubled.
    lt = leading_tone_pc(ctx.key)
    if sum(1 for m in chord if m % 12 == lt) > 1:
        bad.add(5)
    # 7. No doubled 3rd in major-quality triads.
    ctype = info.ctype
    if not ctype.is_seventh and ctype.quality == "major":
        if sum(1 for m in chord if m % 12 == ctype.third_pc) > 1:
            bad.add(7)
    return frozenset(bad)


def static_violations(ctx: BeatContext, chord) -> frozenset[int]:
    """Rules broken by the chord on its own (3, 4 crossing, 5 doubling, 7)."""
    return _static(ctx, tuple(int(m) for m in chord))


def _dominant_pcs(key) -> tuple[frozenset[int], frozenset[int]]:
    """Pitch classes of V and V7 of a key."""
    t = key.lof
    v = frozenset(lof_to_pc(i) for i in (t + 1, t + 5, t + 2))
    return v, v | {lof_to_pc(t - 1)}


def transition_violations(
    prev_ctx: BeatContext | None,
    prev_chord,
    prev_prev_bass: int | None,
    ctx: BeatContext,
    chord,
) -> frozenset[int]:
    """Rules broken by moving from the previous chord to this one.

    With no previous chord only the entry condition of rule 9 can fail.
    """
    chord = tuple(int(m) for m in chord)
    info = _need(ctx, chord)
    bad = set()

    if prev_chord is None:
        # 9. A second-inversion triad with nothing before it can only be cadential.
        if info.is_second_inversion_triad and info.ctype.degree != 1:
            bad.add(9)
        return frozenset(bad)

    prev = tuple(int(m) for m in prev_chord)
    pinfo = _need(prev_ctx, prev)
    move = tuple(c - p for p, c in zip(prev, chord))

    # 1. Parallel perfect 5ths and octaves/unisons, when both voices move.
    for i, j in _PAIRS:
        if move[i] != 0 and move[j] != 0:
            before = (prev[i] - prev[j]) % 12
            after = (chord[i] - chord[j]) % 12
            if before == after and after in (0, 7):
                bad.add(1)
                break

    # 2. Hidden 5ths and octaves into the outer voices.
    if move[0] * move[3] > 0 and abs(move[0]) > 2 and (chord[0] - chord[3]) % 12 in (0, 7):
        bad.add(2)

    # 4. Overlap between neighbouring voices.
    for hi, lo in ((0, 1), (1, 2), (2, 3)):
        if chord[lo] > prev[hi] or chord[hi] < prev[lo]:
            bad.add(4)
            break

    # 5. A leading tone in the bass of a dominant-function chord rises a semitone.
    if (
        pinfo.ctype.dominant
        and prev[3] % 12 == leading_tone_pc(prev_ctx.key)
        and move[3] != 1
    ):
        bad.add(5)

    # 6. The 7th of a seventh chord in A, T or B resolves down by step.
    if pinfo.ctype.is_seventh:
        seventh = pinfo.ctype.seventh_pc
        for v in _LOWER:
            if prev[v] % 12 == seventh and move[v] not in (-1, -2):
                bad.add(6)
                break

    # 8. Alto and tenor leaps at most a perfect 4th.
    if abs(move[1]) > MAX_INNER_LEAP or abs(move[2]) > MAX_INNER_LEAP:
        bad.add(8)

    # 9. Second-inversion triads only as cadential or passing.
    bass_step = _is_step(prev[3], chord[3])
    if info.is_second_inversion_triad and info.ctype.degree != 1 and bass_step == 0:
        bad.add(9)  # not the tonic 6/4, and not approached by step
    if pinfo.is_second_inversion_triad:
        v_pcs, v7_pcs = _dominant_pcs(prev_ctx.key)
        cadential = (
            pinfo.ctype.degree == 1
            and chord[3] == prev[3]
            and frozenset(info.ctype.pcs) in (v_pcs, v7_pcs)
        )
        passing = (
            prev_prev_bass is not None
            and bass_step != 0
            and _is_step(prev_prev_bass, prev[3]) == bass_step
        )
        if not (cadential or passing):
            bad.add(9)

    # 10. No augmented 2nd between the lowered 6th and the raised 7th in minor.
    if ctx.key.mode == "minor":
        low6 = (ctx.key.tonic_pc + 8) % 12
        high7 = (ctx.key.tonic_pc + 11) % 12
        for v in _LOWER:
            if abs(move[v]) == 3 and {prev[v] % 12, chord[v] % 12} == {low6, high7}:
                bad.add(10)
                break

    # 11. No cross-relation. Same letter, different accidental, in different
    #     voices, unless the first voice makes the chromatic step itself.
    #     In line-of-fifths terms: same letter means equal modulo 7.
    hit = False
    for x in range(4):
        px = pinfo.lofs[x]
        for y in range(4):
            cy = info.lofs[y]
            if y == x or cy == px or (cy - px) % 7 != 0:
                continue
            chromatic = info.lofs[x] == cy and abs(move[x]) == abs(cy - px) // 7
            if not chromatic:
                hit = True
                break
        if hit:
            bad.add(11)
            break

    return frozenset(bad)


def violations(
    ctx: BeatContext,
    chord,
    prev_ctx: BeatContext | None = None,
    prev_chord=None,
    prev_prev_bass: int | None = None,
) -> frozenset[int]:
    """Every rule the chord breaks at this beat, given the state before it."""
    return static_violations(ctx, chord) | transition_violations(
        prev_ctx, prev_chord, prev_prev_bass, ctx, chord
    )


def is_legal(
    ctx: BeatContext,
    chord,
    prev_ctx: BeatContext | None = None,
    prev_chord=None,
    prev_prev_bass: int | None = None,
    disabled: frozenset[int] = NO_RULES,
) -> bool:
    if analyse(ctx, chord) is None:
        return False
    return not (violations(ctx, chord, prev_ctx, prev_chord, prev_prev_bass) - disabled)


def legal_candidates(
    ctx: BeatContext,
    prev_ctx: BeatContext | None = None,
    prev_chord=None,
    prev_prev_bass: int | None = None,
    *,
    incremental: bool = True,
    disabled: frozenset[int] = NO_RULES,
) -> list[tuple[int, int, int, int]]:
    """Legal chords at a beat for a given state, in canonical order.

    incremental=True generates from the previous chord (A and T within a
    perfect 4th). incremental=False filters every voicing by brute force.
    Both give the same list.
    """
    if incremental:
        max_leap = None if 8 in disabled else MAX_INNER_LEAP
        pool = incremental_voicings(ctx, prev_chord, max_leap)
    else:
        pool = brute_force_voicings(ctx)
    out = [
        c
        for c in pool
        if not (violations(ctx, c, prev_ctx, prev_chord, prev_prev_bass) - disabled)
    ]
    out.sort(key=canonical_key)
    return out


# ---------------------------------------------------------------------------
# Table form of the same rules, for whole beats at once (numpy).
# The scalar functions above are the reference. Tests check both agree.
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402

from .chords import is_home_tonic  # noqa: E402


class VoicingTable:
    """Every voicing of a beat that passes the static rules, with rule data as arrays."""

    def __init__(self, ctx: BeatContext, disabled: frozenset[int] = NO_RULES):
        self.ctx = ctx
        chords = [
            v for v in brute_force_voicings(ctx) if not (static_violations(ctx, v) - disabled)
        ]
        self.chords = tuple(chords)  # canonical order
        self.index = {c: i for i, c in enumerate(self.chords)}
        n = len(chords)
        self.n = n
        infos = [analyse(ctx, c) for c in chords]
        self.arr = np.array(chords, dtype=np.int32).reshape(n, 4)
        self.lofs = np.array([i.lofs for i in infos], dtype=np.int32).reshape(n, 4)
        self.is64 = np.array([i.is_second_inversion_triad for i in infos], dtype=bool)
        self.tonic64 = np.array(
            [i.is_second_inversion_triad and i.ctype.degree == 1 for i in infos], dtype=bool
        )
        lt = leading_tone_pc(ctx.key)
        self.needs_rise = np.array(
            [i.ctype.dominant and c[3] % 12 == lt for i, c in zip(infos, chords)], dtype=bool
        )
        self.seventh = np.array(
            [
                [i.ctype.is_seventh and m % 12 == i.ctype.seventh_pc for m in c]
                for i, c in zip(infos, chords)
            ],
            dtype=bool,
        ).reshape(n, 4)
        self.typemask = np.array(
            [sum(1 << pc for pc in i.ctype.pcs) for i in infos], dtype=np.int32
        )
        self.root_triad = np.array(
            [i.inversion == 0 and not i.ctype.is_seventh for i in infos], dtype=bool
        )
        self.degree = [i.ctype.degree for i in infos]
        self.home_plain = np.array(
            [is_home_tonic(ctx, i.ctype, picardy=False) for i in infos], dtype=bool
        )
        self.home_picardy = np.array(
            [is_home_tonic(ctx, i.ctype, picardy=True) for i in infos], dtype=bool
        )
        # With nothing before it, a second-inversion triad must be the tonic 6/4 (rule 9).
        self.entry_ok = ~(self.is64 & ~self.tonic64) if 9 not in disabled else np.ones(n, dtype=bool)


@lru_cache(maxsize=4096)
def voicing_table(ctx: BeatContext, disabled: frozenset[int] = NO_RULES) -> VoicingTable:
    static_off = frozenset(disabled & {3, 4, 5, 7, 9})
    return VoicingTable(ctx, static_off)


def transition_tables(prev: VoicingTable, cur: VoicingTable):
    """Rule violations for every (previous voicing, voicing) pair of two beats.

    Returns (bad, step, cond):
      bad[i, j]  uint16 bit set, bit r for a violation of rule r that does not
                 depend on the bass two beats back;
      step[i, j] +1, -1 or 0: the bass steps up, down, or does neither;
      cond[i, j] True where rule 9 passes only as a passing 6/4, i.e. only if
                 the bass also arrived at chord i by a step in the same direction.
    """
    P, C = prev.arr, cur.arr
    n, m = prev.n, cur.n
    move = [C[None, :, v] - P[:, None, v] for v in range(4)]
    bad = np.zeros((n, m), dtype=np.uint16)

    def flag(rule, where):
        np.bitwise_or(bad, np.uint16(1 << rule), out=bad, where=where)

    # 1. Parallel 5ths and octaves.
    r1 = np.zeros((n, m), dtype=bool)
    for i, j in _PAIRS:
        before = (P[:, i] - P[:, j]) % 12
        after = (C[:, i] - C[:, j]) % 12
        perfect = (after == 0) | (after == 7)
        r1 |= (move[i] != 0) & (move[j] != 0) & (before[:, None] == after[None, :]) & perfect[None, :]
    flag(1, r1)

    # 2. Hidden 5ths and octaves into the outer voices.
    outer = (C[:, 0] - C[:, 3]) % 12
    flag(2, (move[0] * move[3] > 0) & (np.abs(move[0]) > 2) & ((outer == 0) | (outer == 7))[None, :])

    # 4. Overlap.
    r4 = np.zeros((n, m), dtype=bool)
    for hi, lo in ((0, 1), (1, 2), (2, 3)):
        r4 |= (C[None, :, lo] > P[:, None, hi]) | (C[None, :, hi] < P[:, None, lo])
    flag(4, r4)

    # 5. Leading tone in the bass of a dominant chord rises a semitone.
    flag(5, prev.needs_rise[:, None] & (move[3] != 1))

    # 6. Sevenths in A, T, B resolve down by step.
    r6 = np.zeros((n, m), dtype=bool)
    for v in _LOWER:
        r6 |= prev.seventh[:, v][:, None] & ~((move[v] == -1) | (move[v] == -2))
    flag(6, r6)

    # 8. Inner leaps.
    flag(8, (np.abs(move[1]) > MAX_INNER_LEAP) | (np.abs(move[2]) > MAX_INNER_LEAP))

    # 9. Second-inversion triads.
    step = np.zeros((n, m), dtype=np.int8)
    step[(move[3] == 1) | (move[3] == 2)] = 1
    step[(move[3] == -1) | (move[3] == -2)] = -1
    entering = (cur.is64 & ~cur.tonic64)[None, :] & (step == 0)
    v_pcs, v7_pcs = _dominant_pcs(prev.ctx.key)
    v_mask = sum(1 << pc for pc in v_pcs)
    v7_mask = sum(1 << pc for pc in v7_pcs)
    is_dominant = (cur.typemask == v_mask) | (cur.typemask == v7_mask)
    cadential = prev.tonic64[:, None] & (move[3] == 0) & is_dominant[None, :]
    leaving = prev.is64[:, None] & ~cadential
    flag(9, entering | (leaving & (step == 0)))
    cond = leaving & (step != 0)

    # 10. Augmented 2nd in minor.
    if cur.ctx.key.mode == "minor":
        low6 = (cur.ctx.key.tonic_pc + 8) % 12
        high7 = (cur.ctx.key.tonic_pc + 11) % 12
        r10 = np.zeros((n, m), dtype=bool)
        for v in _LOWER:
            ppc = (P[:, v] % 12)[:, None]
            cpc = (C[:, v] % 12)[None, :]
            between = ((ppc == low6) & (cpc == high7)) | ((ppc == high7) & (cpc == low6))
            r10 |= (np.abs(move[v]) == 3) & between
        flag(10, r10)

    # 11. Cross-relation.
    r11 = np.zeros((n, m), dtype=bool)
    for x in range(4):
        px = prev.lofs[:, x][:, None]
        own = cur.lofs[:, x][None, :]
        for y in range(4):
            if y == x:
                continue
            cy = cur.lofs[:, y][None, :]
            diff = cy - px
            related = (diff != 0) & (diff % 7 == 0)
            chromatic = (own == cy) & (np.abs(move[x]) == np.abs(diff) // 7)
            r11 |= related & ~chromatic
    flag(11, r11)

    return bad, step, cond
