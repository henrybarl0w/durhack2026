"""start_state, options(), opening chord, cadences with backtracking, pruning.

Everything about one chorale and one weights function lives in a Plan:
the statically legal voicings of every beat, the transition tables between
neighbouring beats, the chosen cadence chords, the opening chord and the
exact "alive" masks used for backward pruning.

State. The spec's state is (prev_chord, prev_prev_bass). The bass two beats
back matters only for rule 9, and only through one fact: whether the bass
reached prev_chord by a step up, a step down, or neither. That is the
"arrival" below (1, 2, 0). It is derived from the exact state, so nothing is
approximated.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .chords import B_RANGE, A_RANGE, T_RANGE, beat_context, canonical_key
from .corpus import Tune, load_tune
from .errors import ChoraleRejected
from .keys import home_tonic_pcs
from .rules import NO_RULES, transition_tables, voicing_table
from .weights import WeightState, WeightsFn, checked_weights, default_weights

TOP_K = 4
_FUNCTION_RANK = {1: 0, 5: 1, 4: 2, 6: 3}  # tonic > dominant > subdominant > submediant
_OTHER_RANK = 4
_ALL_BITS = sum(1 << r for r in range(1, 12))


def _arrival(prev_prev_bass, prev_bass) -> int:
    """0 = no step, 1 = the bass stepped up into the chord, 2 = stepped down."""
    if prev_prev_bass is None:
        return 0
    d = prev_bass - prev_prev_bass
    if d in (1, 2):
        return 1
    if d in (-1, -2):
        return 2
    return 0


class Plan:
    """Cadences, opening chord and pruning masks for one chorale."""

    def __init__(self, tune: Tune, weights: WeightsFn = default_weights,
                 disabled: frozenset[int] = NO_RULES):
        self.tune = tune
        self.chorale_id = tune.chorale_id
        self.weights = weights
        self.disabled = frozenset(disabled)
        self._mask = np.uint16(_ALL_BITS & ~sum(1 << r for r in self.disabled))
        self.tables = [
            voicing_table(beat_context(tune, i), self.disabled) for i in range(len(tune.beats))
        ]
        self._legal_cache: dict[int, tuple] = {}
        self._reach_cache: dict[tuple, np.ndarray | None] = {}
        self._solve_cache: dict[tuple, tuple | None] = {}
        self._options_cache: dict[tuple, tuple] = {}
        self.stuck: tuple[int, int] = (0, 0)  # deepest (phrase, beat) where the search died

        first = tune.beats[0]
        home = tune.home_key
        plain = {home.tonic_pc, (home.tonic_pc + (3 if home.is_minor else 4)) % 12,
                 (home.tonic_pc + 7) % 12}
        self.fixed_opening = first.soprano_midi % 12 in plain

        self.reason: str | None = None
        self.cadences: tuple | None = None  # chord of every phrase's cadence beat
        self.cadence_index: tuple | None = None
        self.opening = None
        self.alive: list[np.ndarray | None] = [None] * len(tune.beats)

        solution = self._solve(0, None)
        if solution is None:
            p, b = self.stuck
            self.reason = f"no valid cadence assignment (no legal path at phrase {p}, beat {b})"
            return
        self.cadence_index = solution
        self.cadences = tuple(
            self.tables[tune.phrases[p][1] - 1].chords[i] for p, i in enumerate(solution)
        )
        for p in range(tune.n_phrases):
            self._prune(p)
        if self.fixed_opening:
            self.opening = self._choose_opening()

    # -- transitions -------------------------------------------------------

    def legal(self, t: int):
        """Legality of moving from beat t-1 to beat t.

        Returns (gen, up, down, sidx). gen[i, j]: legal whatever the arrival
        at i. up/down[i, j]: legal only if the bass stepped up/down into i
        (a passing 6/4). sidx[i, j]: arrival of chord j after this move.
        """
        got = self._legal_cache.get(t)
        if got is None:
            bad, step, cond = transition_tables(self.tables[t - 1], self.tables[t])
            ok = (bad & self._mask) == 0
            if 9 in self.disabled:
                cond = np.zeros_like(cond)
            gen = ok & ~cond
            up = ok & cond & (step == 1)
            down = ok & cond & (step == -1)
            sidx = (step.astype(np.int64) % 3)  # 0, 1, 2 (-1 -> 2)
            got = (gen, up, down, sidx)
            self._legal_cache[t] = got
        return got

    def _advance(self, t: int, reach: np.ndarray) -> np.ndarray:
        """States reachable at beat t from the states reachable at t-1."""
        gen, up, down, sidx = self.legal(t)
        used = (
            (reach[:, 0][:, None] & gen)
            | (reach[:, 1][:, None] & (gen | up))
            | (reach[:, 2][:, None] & (gen | down))
        )
        out = np.zeros((self.tables[t].n, 3), dtype=bool)
        for s in range(3):
            out[:, s] = (used & (sidx == s)).any(axis=0)
        return out

    def _retreat(self, t: int, alive_next: np.ndarray) -> np.ndarray:
        """States at beat t-1 with a legal path onward, from the alive states at t."""
        gen, up, down, sidx = self.legal(t)
        cols = np.arange(self.tables[t].n)[None, :]
        onward = alive_next[cols, sidx]
        out = np.zeros((self.tables[t - 1].n, 3), dtype=bool)
        out[:, 0] = (gen & onward).any(axis=1)
        out[:, 1] = ((gen | up) & onward).any(axis=1)
        out[:, 2] = ((gen | down) & onward).any(axis=1)
        return out

    def _first_states(self, p: int, prev_index: int | None) -> np.ndarray:
        """Reachable states at the first beat of phrase p."""
        start = self.tune.phrases[p][0]
        table = self.tables[start]
        out = np.zeros((table.n, 3), dtype=bool)
        if p == 0:
            if self.fixed_opening:
                out[:, 0] = table.home_plain & table.root_triad
            else:
                out[:, 0] = table.entry_ok
            return out
        gen, _up, _down, sidx = self.legal(start)
        row = gen[prev_index]  # the cadence chord before is never a 6/4
        for s in range(3):
            out[:, s] = row & (sidx[prev_index] == s)
        return out

    def _reach(self, p: int, prev_index: int | None) -> np.ndarray | None:
        """Chords reachable at the cadence beat of phrase p, or None if a beat dies."""
        key = (p, prev_index)
        if key in self._reach_cache:
            return self._reach_cache[key]
        start, end = self.tune.phrases[p]
        reach = self._first_states(p, prev_index)
        result = None
        for t in range(start, end):
            if t > start:
                reach = self._advance(t, reach)
            if not reach.any():
                self.stuck = max(self.stuck, (p, t - start))
                break
        else:
            result = reach.any(axis=1)
        self._reach_cache[key] = result
        return result

    # -- cadences ----------------------------------------------------------

    def _weigh(self, p: int, beat, prev_chord, prev_prev_bass, chords) -> list[int]:
        state = WeightState(
            chorale_id=self.chorale_id,
            phrase_index=p,
            beat_index=beat.beat_index,
            key=beat.key,
            soprano_midi=beat.soprano_midi,
            prev_chord=prev_chord,
            prev_prev_bass=prev_prev_bass,
        )
        return checked_weights(self.weights, state, chords)

    def cadence_candidates(self, p: int, prev_index: int | None) -> list[int]:
        """Ranked cadence candidates of phrase p (voicing indices), best first."""
        reach = self._reach(p, prev_index)
        start, end = self.tune.phrases[p]
        if reach is None:
            return []
        t = end - 1
        table = self.tables[t]
        last = p == self.tune.n_phrases - 1
        if last:
            ok = reach & table.root_triad & (table.home_plain | table.home_picardy)
        else:
            ok = reach & table.root_triad
        idx = [int(i) for i in np.nonzero(ok)[0]]
        if not idx:
            self.stuck = max(self.stuck, (p, t - start))
            return []
        chords = [table.chords[i] for i in idx]
        prev_chord = None
        if p > 0:
            prev_chord = self.tables[start - 1].chords[prev_index]
        ws = self._weigh(p, self.tune.beats[t], prev_chord, None, chords)

        def rank(k):
            i = idx[k]
            if last:
                function = 0 if table.home_picardy[i] else 1  # Picardy third preferred
            else:
                function = _FUNCTION_RANK.get(table.degree[i], _OTHER_RANK)
            return (function, -ws[k], canonical_key(chords[k]))

        return [idx[k] for k in sorted(range(len(idx)), key=rank)]

    def _solve(self, p: int, prev_index: int | None):
        """Cadence indices for phrases p.., highest-ranked first, by backtracking."""
        key = (p, prev_index)
        if key in self._solve_cache:
            return self._solve_cache[key]
        result = None
        for cand in self.cadence_candidates(p, prev_index):
            if p == self.tune.n_phrases - 1:
                result = (cand,)
                break
            rest = self._solve(p + 1, cand)
            if rest is not None:
                result = (cand,) + rest
                break
        self._solve_cache[key] = result
        return result

    # -- pruning -----------------------------------------------------------

    def _prune(self, p: int) -> None:
        """alive[t][j, arrival]: a legal path from that state to C_p exists."""
        start, end = self.tune.phrases[p]
        t = end - 1
        alive = np.zeros((self.tables[t].n, 3), dtype=bool)
        alive[self.cadence_index[p], :] = True
        self.alive[t] = alive
        for t in range(end - 1, start, -1):
            self.alive[t - 1] = self._retreat(t, self.alive[t])

    def _choose_opening(self):
        table = self.tables[0]
        ok = table.home_plain & table.root_triad & self.alive[0][:, 0]
        chords = [table.chords[int(i)] for i in np.nonzero(ok)[0]]
        bach = self.tune.bach_opening
        if bach is not None and tuple(bach) in chords:
            return tuple(bach)

        def distance(c):
            return (
                abs(2 * c[1] - (A_RANGE[0] + A_RANGE[1]))
                + abs(2 * c[2] - (T_RANGE[0] + T_RANGE[1]))
                + abs(2 * c[3] - (B_RANGE[0] + B_RANGE[1]))
            )

        return min(chords, key=lambda c: (distance(c), canonical_key(c)))

    # -- options -----------------------------------------------------------

    def start_state(self, p: int):
        if not 0 <= p < self.tune.n_phrases:
            raise IndexError(f"chorale {self.chorale_id} has no phrase {p}")
        if p == 0:
            return (None, None)
        return (self.cadences[p - 1], None)

    def candidates(self, p: int, b: int, prev_chord, prev_prev_bass) -> list:
        """Legal, unpruned candidates at a decision beat, in canonical order."""
        t = self.tune.phrases[p][0] + b
        table = self.tables[t]
        if t == 0:
            if prev_chord is not None:
                raise ValueError("the first beat of the piece has no previous chord")
            ok = table.entry_ok & self.alive[0][:, 0]
        else:
            if prev_chord is None:
                raise ValueError(f"phrase {p}, beat {b} needs a previous chord")
            i = self.tables[t - 1].index.get(tuple(prev_chord))
            if i is None:
                raise ValueError(
                    f"{tuple(prev_chord)} is not a legal chord at the beat before phrase {p}, beat {b}"
                )
            gen, up, down, sidx = self.legal(t)
            arrival = _arrival(prev_prev_bass, prev_chord[3])
            row = gen[i]
            if arrival == 1:
                row = row | up[i]
            elif arrival == 2:
                row = row | down[i]
            ok = row & self.alive[t][np.arange(table.n), sidx[i]]
        return [table.chords[int(j)] for j in np.nonzero(ok)[0]]

    def options(self, p: int, b: int, prev_chord, prev_prev_bass) -> tuple:
        """The pruned, top-k, ordered option list at a beat."""
        start, end = self.tune.phrases[p]
        if not 0 <= b < end - start:
            raise IndexError(f"phrase {p} has no beat {b}")
        t = start + b
        if t == end - 1:
            return (self.cadences[p],)
        if t == 0 and self.fixed_opening:
            return (self.opening,)
        prev = None if prev_chord is None else tuple(int(m) for m in prev_chord)
        if self.weights is default_weights and prev is not None:
            # The default weights ignore the bass two beats back, so states
            # with the same arrival share one option list.
            key = (t, prev, _arrival(prev_prev_bass, prev[3]))
        else:
            key = (t, prev, prev_prev_bass)
        got = self._options_cache.get(key)
        if got is None:
            cands = self.candidates(p, b, prev, prev_prev_bass)
            ws = self._weigh(p, self.tune.beats[t], prev, prev_prev_bass, cands)
            order = sorted(range(len(cands)), key=lambda k: (-ws[k], canonical_key(cands[k])))
            got = tuple(cands[k] for k in order[:TOP_K])
            self._options_cache[key] = got
        return got


@lru_cache(maxsize=32)
def _plan(chorale_id: int, weights: WeightsFn) -> Plan:
    return Plan(load_tune(chorale_id), weights)


def get_plan(chorale_id: int, weights: WeightsFn = default_weights) -> Plan:
    """The Plan of a usable chorale. Raises ChoraleRejected otherwise."""
    plan = _plan(chorale_id, weights)
    if plan.reason is not None:
        raise ChoraleRejected(chorale_id, plan.reason)
    return plan


def plan_reason(chorale_id: int, weights: WeightsFn = default_weights) -> str | None:
    """Why the chorale cannot be harmonized, or None if it can."""
    try:
        get_plan(chorale_id, weights)
    except ChoraleRejected as exc:
        return exc.reason
    return None


def start_state(chorale_id: int, phrase_index: int, weights: WeightsFn = default_weights):
    """(prev_chord, prev_prev_bass) at the first beat of a phrase."""
    return get_plan(chorale_id, weights).start_state(phrase_index)


def options(chorale_id: int, phrase_index: int, beat_index: int, prev_chord, prev_prev_bass,
            weights: WeightsFn = default_weights) -> list:
    """Exactly the pruned, top-k, ordered option list the encoder uses at a beat.

    [C_p] at a cadence beat and [opening chord] at a fixed opening beat.
    """
    return list(
        get_plan(chorale_id, weights).options(phrase_index, beat_index, prev_chord, prev_prev_bass)
    )


# ---------------------------------------------------------------------------
# Which chorales are usable (default weights)
# ---------------------------------------------------------------------------

def _source_digest() -> str:
    """Hash of the engine source, so the on-disk list never outlives a code change."""
    import hashlib
    from pathlib import Path

    h = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


@lru_cache(maxsize=1)
def _status() -> dict[int, str | None]:
    """Rejection reason (or None) for every Riemenschneider number."""
    import json
    import os

    from . import corpus

    path = corpus._cache_dir() / f"status-{_source_digest()}.json"
    if path.is_file():
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return {int(k): v for k, v in json.load(fh).items()}
        except (OSError, ValueError):
            pass
    status: dict[int, str | None] = {}
    for cid in corpus.chorale_ids():
        reason = corpus.filter_reason(cid)
        if reason is None:
            reason = Plan(load_tune(cid)).reason
        status[cid] = reason
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({str(k): v for k, v in sorted(status.items())}, fh, sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass
    return status


def usable_chorales() -> list[int]:
    """Chorale IDs the engine can encode into, ascending."""
    return [cid for cid, reason in sorted(_status().items()) if reason is None]


def rejected_chorales() -> dict[int, str]:
    """Rejected chorale IDs with the reason."""
    return {cid: reason for cid, reason in sorted(_status().items()) if reason is not None}
