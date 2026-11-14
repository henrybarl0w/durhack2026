"""Survey every chorale in the music21 Riemenschneider list.

Per chorale: usable or the rejection reason, phrases, beats, bits, the beat
with no legal path (if any) and the rule blame: which of rules 1-11, when
switched off alone, would make the chorale usable.

    python scripts/survey.py                # all chorales
    python scripts/survey.py --ids 6 16 206
    python scripts/survey.py --summary-only
"""

from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chorale_engine import corpus  # noqa: E402
from chorale_engine.options import Plan  # noqa: E402

try:  # capacity arrives with the codec
    from chorale_engine.codec import capacity_report  # noqa: E402
except ImportError:  # pragma: no cover
    capacity_report = None

RULES = tuple(range(1, 12))


def blame(tune) -> list[int]:
    """Rules whose removal alone makes the chorale usable."""
    return [r for r in RULES if Plan(tune, disabled=frozenset({r})).reason is None]


def survey_one(cid: int, with_blame: bool = True) -> dict:
    row = {"id": cid, "status": "usable", "phrases": None, "beats": None, "bits": None,
           "stuck": None, "blame": None, "seconds": 0.0}
    reason = corpus.filter_reason(cid)
    if reason is not None:
        row["status"] = f"filtered: {reason}"
        return row
    tune = corpus.load_tune(cid)
    row["phrases"] = tune.n_phrases
    row["beats"] = len(tune.beats)
    t0 = time.perf_counter()
    plan = Plan(tune)
    if plan.reason is not None:
        row["status"] = "no legal path"
        row["stuck"] = plan.stuck
        if with_blame:
            row["blame"] = blame(tune)
    elif capacity_report is not None:
        row["bits"] = capacity_report(cid)["bits"]
    row["seconds"] = time.perf_counter() - t0
    return row


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ids", type=int, nargs="*", help="chorale IDs (default: all)")
    parser.add_argument("--no-blame", action="store_true", help="skip the rule blame reruns")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    ids = args.ids if args.ids else list(corpus.chorale_ids())
    rows = []
    for cid in ids:
        row = survey_one(cid, with_blame=not args.no_blame)
        rows.append(row)
        if not args.summary_only:
            extra = ""
            if row["stuck"] is not None:
                extra = f"  dead at phrase {row['stuck'][0]} beat {row['stuck'][1]}"
                if row["blame"] is not None:
                    extra += f"  fixed by removing rule: {row['blame'] or 'none alone'}"
            bits = "-" if row["bits"] is None else row["bits"]
            print(f"{cid:4d}  {row['status']:<60s} phrases={row['phrases'] or '-'} "
                  f"beats={row['beats'] or '-'} bits={bits}{extra}")

    passed_filter = [r for r in rows if not r["status"].startswith("filtered")]
    usable = [r for r in rows if r["status"] == "usable"]
    dead = [r for r in rows if r["status"] == "no legal path"]
    print()
    print("SUMMARY")
    print(f"  music21 {corpus.MUSIC21_VERSION}")
    print(f"  chorales surveyed:            {len(rows)}")
    print(f"  pass the corpus filter:       {len(passed_filter)}")
    print(f"  usable:                       {len(usable)}")
    share = 100 * len(dead) / len(passed_filter) if passed_filter else 0.0
    print(f"  a beat with no legal path:    {len(dead)} ({share:.1f}% of those passing the filter)")
    reasons = collections.Counter(
        r["status"].split(" (")[0] for r in rows if r["status"].startswith("filtered")
    )
    for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:4d}  {reason}")
    if dead:
        print("  no legal path:")
        for r in dead:
            line = f"    chorale {r['id']}: phrase {r['stuck'][0]}, beat {r['stuck'][1]}"
            if r["blame"] is not None:
                line += f"; fixed by removing rule {r['blame'] or 'none alone'}"
            print(line)
        if any(r["blame"] is not None for r in dead):
            fixes = collections.Counter(rule for r in dead for rule in (r["blame"] or []))
            print("  rule blame (chorales fixed by removing that rule alone):")
            for rule in RULES:
                print(f"    rule {rule:2d}: {fixes.get(rule, 0)}")
    if usable and usable[0]["bits"] is not None:
        bits = sorted(r["bits"] for r in usable)
        print(f"  bits per usable chorale: min {bits[0]}, median {bits[len(bits) // 2]}, max {bits[-1]}")
    if passed_filter:
        slowest = max(passed_filter, key=lambda r: r["seconds"])
        print(f"  slowest chorale: {slowest['id']} ({slowest['seconds']:.2f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
