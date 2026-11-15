"""Experimental strict style: textbook chorale rules on top of the locked hard rules.

The number being encoded picks among the offered chords at random, so a rule that
only lowers a chord's weight still lets it through whenever it is among the top
options. These rules remove chords from the options instead. They are opt-in:

    from chorale_engine.style import strict_weights
    chords = encode(cid, n, strict_weights)

Rules (StyleRules sets which ones remove options; the rest are strong preferences in strict_weights):
  - doubling: root position doubles the root (a tripled root with no 5th is fine);
    first inversion doubles the root or the 3rd (a diminished triad: the 3rd or 5th);
    second inversion doubles the 5th, the bass; a seventh chord doubles nothing, or the
    root when the 5th is left out;
  - bass leaps: at most a perfect 5th, or an octave; never a tritone, 6th or 7th;
  - exposed 5ths and octaves: no two voices move the same way into a 5th or octave
    while the upper of the two leaps (the locked rule 2 covers soprano and bass only);
  - harmony: no chord change that Bach makes less than 0.3% of the time after the chord
    before (from the Bach model's chord-after-chord table).
  - cadences: a phrase that ends on the tonic arrives from V, V7 or vii (an authentic
    cadence, not IV-I), where the tune allows;
  - ending: the chord before the final cadence has dominant function (V, V7, vii), and
    root-position V or V7 is preferred there: a perfect cadence;
  - contrary motion: a bonus for the bass moving against the soprano.

The rules only remove options, so every remaining path is still legal under rules 1-11.
A chorale where they leave no path to a cadence cannot be used with strict_weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy

from .bach_weights import bach_weights, chord_class, load_model, sign
from .chords import analyse

ALLOWED_BASS_LEAPS = frozenset({0, 1, 2, 3, 4, 5, 7, 12})


@dataclass(frozen=True)
class StyleRules:
    """Which style rules remove options outright. The defaults are the ones the ear notices most
    (big bass leaps, odd chord changes); doubling and exposed 5ths cost a lot of capacity as hard rules,
    so strict_weights treats them as strong preferences instead."""

    doubling: bool = False
    bass_leaps: bool = True
    exposed: bool = False
    harmony_floor: float = 0.003  # 0 switches the harmony rule off
    final_dominant: bool = True  # the chord before the final cadence is V, V7 or vii
    cadence_dominant: bool = True  # a phrase that ends on the tonic gets there from V, V7 or vii (no IV-I)


def good_doubling(info) -> bool:
    """The textbook doubling for this voicing."""
    ctype = info.ctype
    pcs = [m % 12 for m in info.chord]
    count = {pc: pcs.count(pc) for pc in set(pcs)}
    tones = ctype.pcs
    if ctype.is_seventh:
        if len(count) == 4:
            return True
        # 5th left out: the root is the doubled note
        return tones[2] not in count and count.get(tones[0], 0) == 2
    root, third, fifth = tones
    doubled = [pc for pc, n in count.items() if n >= 2]
    if info.inversion == 0:
        if fifth not in count:
            return count.get(root, 0) == 3
        return doubled == [root]
    if fifth not in count or root not in count:
        return False
    if info.inversion == 1:
        if ctype.quality == "diminished":
            return doubled[0] in (third, fifth)
        return doubled[0] in (root, third)
    return doubled == [fifth]


@lru_cache(maxsize=4096)
def _static_mask(table, rules: StyleRules) -> numpy.ndarray:
    if not rules.doubling:
        return numpy.ones(table.n, bool)
    return numpy.array([good_doubling(analyse(table.ctx, c)) for c in table.chords], bool)


def static_ok(table, rules: StyleRules) -> numpy.ndarray:
    """Per voicing of a beat: does it pass the style rules on its own?"""
    return _static_mask(table, rules)


def _harmony_mask(prev, cur, floor: float) -> numpy.ndarray:
    model = load_model()["chord"]["after"]
    tonic = cur.ctx.key.tonic_pc  # as in training: both chords read in the later beat's key
    prev_classes = [chord_class(c, tonic) for c in prev.chords]
    cur_classes = [chord_class(c, tonic) for c in cur.chords]
    cutoff = int(math.floor(100 * math.log(floor)))
    ok = numpy.ones((prev.n, cur.n), bool)
    cache = {}
    for i, pc in enumerate(prev_classes):
        row = model.get(pc)
        if row is None:
            continue  # a chord Bach never used: nothing to judge by
        if pc not in cache:
            cache[pc] = numpy.array([row.get(cc, -10 ** 9) >= cutoff for cc in cur_classes], bool)
        ok[i] = cache[pc]
    return ok


@lru_cache(maxsize=1024)
def _dominant_rows(table) -> numpy.ndarray:
    return numpy.array([analyse(table.ctx, c).ctype.dominant for c in table.chords], bool)


def transition_ok(prev, cur, rules: StyleRules, final: bool = False, cadence: bool = False) -> numpy.ndarray:
    """For every (previous voicing, voicing) pair of two beats: does the move pass the style rules?

    final: the later beat is the last chord of the piece. cadence: it ends a phrase.
    """
    P, C = prev.arr, cur.arr
    ok = numpy.ones((prev.n, cur.n), bool)
    if cadence and not final and rules.cadence_dominant:
        # An authentic cadence: a phrase that comes to rest on the tonic arrives from its dominant.
        tonic = numpy.array([d == 1 for d in cur.degree], bool) | cur.home_plain | cur.home_picardy
        same_key = prev.ctx.key == cur.ctx.key
        ok &= ~tonic[None, :] | (_dominant_rows(prev)[:, None] & same_key)
    if final and rules.final_dominant:
        # A perfect cadence to end: the chord before the last one has dominant function.
        ok &= _dominant_rows(prev)[:, None]
    if rules.bass_leaps:
        leap = numpy.abs(C[None, :, 3] - P[:, None, 3])
        ok &= numpy.isin(leap, list(ALLOWED_BASS_LEAPS))
    if rules.exposed:
        for hi in range(4):
            for lo in range(hi + 1, 4):
                if (hi, lo) == (0, 3):
                    continue  # the locked rule 2 already covers soprano and bass
                move_hi = C[None, :, hi] - P[:, None, hi]
                move_lo = C[None, :, lo] - P[:, None, lo]
                interval = ((C[:, hi] - C[:, lo]) % 12)[None, :]
                perfect = (interval == 0) | (interval == 7)
                ok &= ~((numpy.sign(move_hi) == numpy.sign(move_lo)) & (move_hi != 0) & perfect & (numpy.abs(move_hi) > 2))
    if rules.harmony_floor > 0:
        ok &= _harmony_mask(prev, cur, rules.harmony_floor)
    return ok


CONTRARY_BONUS = 150  # points for the bass moving against the soprano, or holding while it moves
DOUBLING_PENALTY = 800  # points lost for a doubling the textbook would not choose
FINAL_V_BONUS = 600  # points for root-position V or V7 just before the final chord
EXPOSED_PENALTY = 600  # points lost for an exposed 5th or octave between any other pair of voices


def _exposed(prev, chord) -> bool:
    for hi in range(4):
        for lo in range(hi + 1, 4):
            if (hi, lo) == (0, 3):
                continue
            mh, ml = chord[hi] - prev[hi], chord[lo] - prev[lo]
            if mh and sign(mh) == sign(ml) and abs(mh) > 2 and (chord[hi] - chord[lo]) % 12 in (0, 7):
                return True
    return False


def strict_weights(state, options) -> list[int]:
    """The Bach model with textbook preferences on top: contrary motion between soprano and bass,
    textbook doubling, no exposed 5ths or octaves. With StyleRules, big bass leaps and rare chord
    changes are not offered at all."""
    from .chords import BeatContext
    from .corpus import load_tune

    base = bach_weights(state, options)
    tune = load_tune(state.chorale_id)
    index = tune.phrases[state.phrase_index][0] + state.beat_index
    ctx = BeatContext(state.key, state.soprano_midi, tune.home_key, index == 0, index == len(tune.beats) - 1)
    prev = state.prev_chord
    out = []
    penultimate = index == len(tune.beats) - 2
    for w, option in zip(base, options):
        info = analyse(ctx, option)
        if info is not None and not good_doubling(info):
            w -= DOUBLING_PENALTY
        if penultimate and info is not None and info.ctype.degree == 5 and info.ctype.dominant and info.inversion == 0:
            w += FINAL_V_BONUS
        if prev is not None:
            s, b = sign(option[0] - prev[0]), sign(option[3] - prev[3])
            if s != 0 and (b == -s or b == 0):
                w += CONTRARY_BONUS
            if _exposed(prev, option):
                w -= EXPOSED_PENALTY
        out.append(w)
    return out


strict_weights.style = StyleRules()
strict_weights.ignores_prev_prev_bass = True
