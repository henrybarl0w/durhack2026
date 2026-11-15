"""Compare harmonizations: Bach's own, the phase 1 weights and the Bach model.

For alto, tenor and bass: how often a voice repeats its note from one beat to
the next, and how far it moves. Also how often the whole chord is repeated,
and the capacity in bits.

    python scripts/music_metrics.py            # 40 chorales, 3 numbers each
    python scripts/music_metrics.py --chorales 20 --numbers 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chorale_engine as ce  # noqa: E402
from chorale_engine.bach_weights import bach_weights, chord_class, load_model, with_margin  # noqa: E402
from chorale_engine.keys import Key  # noqa: E402
from chorale_engine.style import strict_weights  # noqa: E402
from chorale_engine.weights import default_weights  # noqa: E402

BACH = Path(__file__).resolve().parent.parent / ".cache" / "bach_chords.json"


class Tally:
    def __init__(self):
        self.pairs = 0
        self.repeats = [0, 0, 0]
        self.motion = [0, 0, 0]
        self.same_chord = 0
        self.all_still = 0
        self.leaps4 = [0, 0, 0]  # a perfect 4th or more
        self.leaps6 = [0, 0, 0]  # a minor 6th or more
        self.rare = 0  # chord changes Bach makes less than 1% of the time

    def add(self, chords, cadence, keys=None):
        after = load_model()["chord"]["after"]
        for i in range(1, len(chords)):
            a, b = chords[i - 1], chords[i]
            if a is None or b is None or cadence[i - 1]:
                continue  # do not count across a held cadence chord
            self.pairs += 1
            for v in (1, 2, 3):
                self.repeats[v - 1] += a[v] == b[v]
                self.motion[v - 1] += abs(a[v] - b[v])
                self.leaps4[v - 1] += abs(a[v] - b[v]) >= 5
                self.leaps6[v - 1] += abs(a[v] - b[v]) >= 8
            self.same_chord += tuple(a) == tuple(b)
            if keys is not None:
                tonic = keys[i].tonic_pc
                row = after.get(chord_class(a, tonic))
                if row is not None and row.get(chord_class(b, tonic), -10 ** 9) < -460:
                    self.rare += 1
            self.all_still += all(a[v] == b[v] for v in (1, 2, 3))

    def row(self, name, bits=None):
        p = max(1, self.pairs)
        rep = " ".join(f"{100 * r / p:4.0f}%" for r in self.repeats)
        mot = " ".join(f"{m / p:4.1f}" for m in self.motion)
        extra = "" if bits is None else f"  {bits:6.1f}"
        l4 = " ".join(f"{100 * x / p:3.0f}%" for x in self.leaps4)
        l6 = " ".join(f"{100 * x / p:3.0f}%" for x in self.leaps6)
        return (f"{name:<14s} {rep}   {mot}   {l4}   {l6}   {100 * self.rare / p:4.0f}%"
                f"   {100 * self.all_still / p:4.0f}%{extra}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chorales", type=int, default=40)
    parser.add_argument("--numbers", type=int, default=3)
    parser.add_argument("--margins", type=int, nargs="*", default=[], help="also try the Bach model with these margins")
    args = parser.parse_args(argv)

    bach_data = json.loads(BACH.read_text())
    usable = ce.usable_chorales()
    step = max(1, len(usable) // args.chorales)
    chosen = usable[::step][: args.chorales]
    rng = random.Random(2026)

    bach = Tally()
    for cid in chosen:
        entry = bach_data[str(cid)]
        bach.add(entry["chords"], entry["cadence"], [Key(t, m) for t, m in entry["key"]])

    results = {}
    variants = [("phase 1", default_weights), ("Bach model", bach_weights), ("strict style", strict_weights)]
    variants += [(f"margin {m}", with_margin(m)) for m in args.margins]
    for name, weights in variants:
        tally = Tally()
        bits = []
        lost = []
        for cid in chosen:
            tune = ce.load_tune(cid)
            cadence = [b.is_cadence for b in tune.beats]
            try:
                report = ce.capacity_report(cid, weights)
            except ce.ChoraleRejected:
                lost.append(cid)
                continue
            bits.append(report["bits"])
            for _ in range(args.numbers):
                n = rng.randrange(report["capacity"])
                chords = ce.encode(cid, n, weights)
                assert ce.decode(cid, chords, weights) == n
                tally.add(chords, cadence, [b.key for b in tune.beats])
        results[name] = (tally, sum(bits) / len(bits), lost)

    print(f"{len(chosen)} chorales, {args.numbers} random numbers each")
    print(f"{'':<14s} repeats a note      semitones/beat   leap 4th or more   leap 6th or more   rare     all three   mean")
    print(f"{'':<14s}    A    T    B        A    T    B      A    T    B        A    T    B      chords   still       bits")
    print(bach.row("Bach himself"))
    for name, (tally, mean_bits, lost) in results.items():
        print(tally.row(name, mean_bits) + (f"   ({len(lost)} chorales unusable)" if lost else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
