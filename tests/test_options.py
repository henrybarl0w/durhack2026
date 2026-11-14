"""Milestone 3: options, cadences, opening chord and pruning."""

import functools
import random

import pytest

from chorale_engine import (
    ChoraleRejected,
    load_tune,
    options,
    rejected_chorales,
    start_state,
    usable_chorales,
)
from chorale_engine.chords import (
    A_RANGE,
    B_RANGE,
    T_RANGE,
    analyse,
    beat_context,
    brute_force_voicings,
    canonical_key,
    is_home_tonic,
)
from chorale_engine.options import TOP_K, Plan, get_plan
from chorale_engine.rules import (
    legal_candidates,
    static_violations,
    transition_tables,
    transition_violations,
    voicing_table,
    _is_step,
)
from chorale_engine.weights import WeightState, default_weights

SAMPLE = [6, 8, 14, 16, 21, 47, 58, 175, 227]


def walk(cid, rng=None, weights=default_weights):
    """Follow options() through the piece. Returns [(phrase, beat, state, options, chosen)]."""
    tune = load_tune(cid)
    out = []
    for p in range(tune.n_phrases):
        prev, ppb = start_state(cid, p, weights)
        for b in range(len(tune.phrase_beats(p))):
            opts = options(cid, p, b, prev, ppb, weights)
            chosen = opts[0] if rng is None else rng.choice(opts)
            out.append((p, b, (prev, ppb), opts, chosen))
            prev, ppb = chosen, (prev[3] if prev is not None else None)
    return out


def test_at_least_twenty_usable_chorales():
    usable = usable_chorales()
    assert len(usable) >= 20
    assert set(usable).isdisjoint(rejected_chorales())
    for cid in SAMPLE:
        assert cid in usable


def test_tables_match_the_scalar_rules():
    rng = random.Random(7)
    checked = 0
    for cid in (16, 47, 227):
        tune = load_tune(cid)
        for t in rng.sample(range(1, len(tune.beats)), 2):
            prev_t = voicing_table(beat_context(tune, t - 1))
            cur_t = voicing_table(beat_context(tune, t))
            bad, step, cond = transition_tables(prev_t, cur_t)
            pairs = [(i, j) for i in range(prev_t.n) for j in range(cur_t.n)]
            for i, j in rng.sample(pairs, min(len(pairs), 1500)):
                p, c = prev_t.chords[i], cur_t.chords[j]
                for ppb in (None, p[3] - 1, p[3] + 2, p[3] + 5):
                    expected = transition_violations(prev_t.ctx, p, ppb, cur_t.ctx, c)
                    got = {r for r in range(1, 12) if (int(bad[i, j]) >> r) & 1}
                    arrival = 0 if ppb is None else _is_step(ppb, p[3])
                    if cond[i, j] and arrival != step[i, j]:
                        got.add(9)
                    assert got == set(expected), (cid, t, p, c, ppb)
                    checked += 1
    assert checked > 10000
    # The table holds exactly the statically legal voicings.
    ctx = beat_context(load_tune(16), 3)
    table = voicing_table(ctx)
    assert list(table.chords) == [v for v in brute_force_voicings(ctx) if not static_violations(ctx, v)]


def test_options_are_top_k_in_weight_then_canonical_order():
    rng = random.Random(11)
    for cid in SAMPLE:
        tune = load_tune(cid)
        plan = get_plan(cid)
        for p, b, (prev, ppb), opts, _chosen in walk(cid, rng):
            beat = tune.beat(p, b)
            assert 1 <= len(opts) <= TOP_K
            assert len(set(opts)) == len(opts)
            if beat.is_cadence:
                assert opts == [plan.cadences[p]]
                continue
            if beat.index == 0 and plan.fixed_opening:
                assert opts == [plan.opening]
                continue
            cands = plan.candidates(p, b, prev, ppb)
            assert cands == sorted(cands, key=canonical_key)
            state = WeightState(cid, p, b, beat.key, beat.soprano_midi, prev, ppb)
            ws = default_weights(state, cands)
            ranked = [c for _w, _k, c in sorted(
                (-w, canonical_key(c), c) for w, c in zip(ws, cands)
            )]
            assert opts == ranked[:TOP_K]
            # Every candidate is legal by the scalar rules.
            ctx = beat_context(tune, beat.index)
            prev_ctx = beat_context(tune, beat.index - 1) if beat.index else None
            legal = legal_candidates(ctx, prev_ctx, prev, ppb)
            assert set(cands) <= set(legal)


def _scalar_alive(tune, plan, p):
    """Reference for the pruning: can this state still reach C_p? Scalar rules only."""
    start, end = tune.phrases[p]
    target = plan.cadences[p]

    @functools.lru_cache(maxsize=None)
    def alive(t, chord, arrival_bass):
        # arrival_bass stands in for the bass two beats back.
        if t == end - 1:
            return chord == target
        nxt = legal_candidates(
            beat_context(tune, t + 1), beat_context(tune, t), chord, arrival_bass
        )
        return any(alive(t + 1, c, chord[3]) for c in nxt)

    return alive


@pytest.mark.parametrize("cid,p", [(14, 2), (19, 3), (86, 3)])
def test_pruning_is_exact_on_short_phrases(cid, p):
    """Candidates == scalar-legal chords that still have a legal path to C_p."""
    tune = load_tune(cid)
    plan = get_plan(cid)
    alive = _scalar_alive(tune, plan, p)
    start, end = tune.phrases[p]
    rng = random.Random(cid)
    for _ in range(3):
        prev, ppb = start_state(cid, p)
        for b in range(end - start - 1):
            t = start + b
            got = plan.candidates(p, b, prev, ppb)
            legal = legal_candidates(beat_context(tune, t), beat_context(tune, t - 1), prev, ppb)
            expected = [c for c in legal if alive(t, c, prev[3])]
            assert got == expected, (cid, p, b)
            assert got, "a reachable state always has an option"
            opts = options(cid, p, b, prev, ppb)
            assert set(opts) <= set(got)
            chosen = rng.choice(opts)
            prev, ppb = chosen, prev[3]
        assert options(cid, p, end - start - 1, prev, ppb) == [plan.cadences[p]]


def test_every_offered_option_has_a_legal_completion():
    # Taking any option at every beat always arrives at the cadence chord by legal moves.
    for seed, cid in enumerate(SAMPLE):
        tune = load_tune(cid)
        plan = get_plan(cid)
        steps = walk(cid, random.Random(seed))
        chords = [chosen for *_rest, chosen in steps]
        assert len(chords) == len(tune.beats)
        for p, b, (prev, ppb), opts, chosen in steps:
            beat = tune.beat(p, b)
            ctx = beat_context(tune, beat.index)
            assert chosen[0] == beat.soprano_midi
            assert not static_violations(ctx, chosen)
            prev_ctx = beat_context(tune, beat.index - 1) if beat.index else None
            assert not transition_violations(prev_ctx, prev, ppb, ctx, chosen), (cid, p, b)
            if beat.is_cadence:
                assert chosen == plan.cadences[p]


def test_start_state():
    for cid in SAMPLE:
        plan = get_plan(cid)
        assert start_state(cid, 0) == (None, None)
        for p in range(1, plan.tune.n_phrases):
            assert start_state(cid, p) == (plan.cadences[p - 1], None)
    with pytest.raises(IndexError):
        start_state(6, 99)


def test_cadences_are_root_position_triads_ranked_by_function():
    for cid in SAMPLE:
        tune = load_tune(cid)
        plan = get_plan(cid)
        assert len(plan.cadences) == tune.n_phrases
        for p, chord in enumerate(plan.cadences):
            t = tune.phrases[p][1] - 1
            info = analyse(beat_context(tune, t), chord)
            assert info.inversion == 0 and not info.ctype.is_seventh
            assert chord[0] == tune.beats[t].soprano_midi
        # Phrase 0: the chosen cadence is the best-ranked candidate that the
        # later phrases can follow, and its function rank is the best available.
        ranked = plan.cadence_candidates(0, None)
        table = plan.tables[tune.phrases[0][1] - 1]
        order = {1: 0, 5: 1, 4: 2, 6: 3}
        ranks = [order.get(table.degree[i], 4) for i in ranked]
        if tune.n_phrases > 1:
            assert ranks == sorted(ranks)
        chosen = plan.cadence_index[0]
        assert chosen in ranked
        for better in ranked[: ranked.index(chosen)]:
            assert plan._solve(1, better) is None  # skipped only because it leads nowhere


def test_final_chord_is_the_home_tonic_in_root_position():
    minor_seen = picardy_seen = 0
    for cid in usable_chorales():
        tune = load_tune(cid)
        plan = get_plan(cid)
        final = plan.cadences[-1]
        home = tune.home_key
        ctx = beat_context(tune, len(tune.beats) - 1)
        info = analyse(ctx, final)
        assert is_home_tonic(ctx, info.ctype)
        assert final[3] % 12 == home.tonic_pc  # root position
        pcs = {m % 12 for m in final}
        major_third = (home.tonic_pc + 4) % 12
        minor_third = (home.tonic_pc + 3) % 12
        if home.mode == "major":
            assert major_third in pcs
            continue
        minor_seen += 1
        # Picardy third whenever it is legal: it is among the candidates
        # exactly when a major-tonic voicing is reachable.
        cands = plan.cadence_candidates(tune.n_phrases - 1, (
            plan.cadence_index[-2] if tune.n_phrases > 1 else None
        ))
        table = plan.tables[len(tune.beats) - 1]
        picardy_legal = any(table.home_picardy[i] for i in cands)
        if picardy_legal:
            assert major_third in pcs and minor_third not in pcs, cid
            picardy_seen += 1
        else:
            assert minor_third in pcs
            # Only possible when the tune itself ends on the minor 3rd.
            assert final[0] % 12 == minor_third, cid
    assert minor_seen > 20 and picardy_seen > 20


def _centre_distance(c):
    return (
        abs(2 * c[1] - sum(A_RANGE)) + abs(2 * c[2] - sum(T_RANGE)) + abs(2 * c[3] - sum(B_RANGE))
    )


def test_opening_chord_rules():
    kinds = {"bach": 0, "centred": 0, "free": 0}
    for cid in usable_chorales():
        tune = load_tune(cid)
        plan = get_plan(cid)
        home = tune.home_key
        triad = {home.tonic_pc, (home.tonic_pc + (3 if home.is_minor else 4)) % 12,
                 (home.tonic_pc + 7) % 12}
        first = options(cid, 0, 0, None, None)
        if tune.beats[0].soprano_midi % 12 not in triad:
            # Not in the home tonic triad: an ordinary decision beat.
            assert not plan.fixed_opening and plan.opening is None
            assert first == list(plan.options(0, 0, None, None))
            kinds["free"] += 1
            continue
        assert plan.fixed_opening and first == [plan.opening]
        ctx = beat_context(tune, 0)
        info = analyse(ctx, plan.opening)
        assert is_home_tonic(ctx, info.ctype, picardy=False) and info.inversion == 0
        assert not static_violations(ctx, plan.opening)
        # Legal root-position tonic voicings with a legal path to C_0.
        table = plan.tables[0]
        legal = [
            c for i, c in enumerate(table.chords)
            if table.home_plain[i] and table.root_triad[i] and plan.alive[0][i, 0]
        ]
        assert plan.opening in legal
        if tune.bach_opening in legal:
            assert plan.opening == tune.bach_opening
            kinds["bach"] += 1
        else:
            best = min(legal, key=lambda c: (_centre_distance(c), canonical_key(c)))
            assert plan.opening == best
            kinds["centred"] += 1
    assert all(n > 0 for n in kinds.values()), kinds


def test_opening_has_a_path_to_the_first_cadence():
    # Check "legal path to C_0" for the opening with the scalar rules, on a short first phrase.
    cid = 48
    tune = load_tune(cid)
    plan = get_plan(cid)
    assert plan.fixed_opening
    alive = _scalar_alive(tune, plan, 0)
    assert alive(0, plan.opening, None)


def test_plan_is_deterministic():
    for cid in (6, 16, 58):
        a = Plan(load_tune(cid))
        b = Plan(load_tune(cid))
        assert a.cadences == b.cadences and a.opening == b.opening
        assert a.cadences == get_plan(cid).cadences
        assert [o for *_x, o, _c in walk(cid)] == [o for *_x, o, _c in walk(cid)]


def test_custom_weights_change_the_options_but_not_the_legality():
    def reversed_weights(state, opts):
        return [-w for w in default_weights(state, opts)]

    cid = 6
    tune = load_tune(cid)
    default_steps = walk(cid)
    custom_steps = walk(cid, weights=reversed_weights)
    assert [c for *_x, c in default_steps] != [c for *_x, c in custom_steps]
    for p, b, (prev, ppb), opts, chosen in custom_steps:
        beat = tune.beat(p, b)
        ctx = beat_context(tune, beat.index)
        prev_ctx = beat_context(tune, beat.index - 1) if beat.index else None
        assert not static_violations(ctx, chosen)
        assert not transition_violations(prev_ctx, prev, ppb, ctx, chosen)
    # A wrapper around the default gives exactly the default's options.
    def same(state, opts):
        return default_weights(state, opts)

    assert [o for *_x, o, _c in walk(cid, weights=same)] == [o for *_x, o, _c in default_steps]


def test_rejected_chorale_raises():
    rejected = rejected_chorales()
    assert rejected[206].startswith("no valid cadence assignment")
    with pytest.raises(ChoraleRejected):
        options(206, 0, 0, None, None)
    with pytest.raises(ChoraleRejected):
        start_state(1, 0)


def test_options_argument_errors():
    with pytest.raises(IndexError):
        options(6, 0, 99, None, None)
    with pytest.raises(ValueError):
        options(6, 0, 1, None, None)  # needs a previous chord
    with pytest.raises(ValueError):
        options(6, 0, 1, (65, 60, 57, 40), None)  # not a legal chord at beat 0
