#which improvement helps, and which one broke something: the reader is run on one saved page with every combination of
#its five steps switched on and off (reader/features.py), each result scored against what the page SHOULD say.
#   python debugablate.py "Hello world"                      the newest scan (debug/latest/scan.png)
#   python debugablate.py "Hello world" --page some.png --chorale 241
#columns: exact notes, wrong pitches, missed, invented, then the change against "all on". a step whose removal RAISES the
#exact count is hurting; one that crashed or broke an invariant is named in the health lines under the table.
#writes ablation.txt next to the page
import argparse
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader import features
from reader.vote import combine, readVoted

from debugcompare import match, truthNotes

steps = ("curves", "keyVote", "columns", "restGuard", "vote")
folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug", "latest")


def run(page, truth, label, **switches):
    trace = {}
    started = time.time()
    with features.using(**switches), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reads = readVoted(page, trace)
        notes, ticks = combine(reads, trace)
    read = [{"pitch": n.pitch, "startBeat": n.start / ticks, "endBeat": n.end / ticks} for n in notes]
    matched, results, invented = match(truth, read)
    counts = {"exact": 0, "wrong pitch": 0, "missed": 0, "wrong length": 0, "wrong time": 0}
    for result in results:
        counts[result[0] if result else "missed"] += 1
    health = []
    for step, entry in trace.get("steps", {}).items():
        if entry["crashed"]:
            health.append(f"{step} CRASHED: {entry['error'].strip().splitlines()[-1]}")
        for violation in entry["violations"][:2]:
            health.append(f"{step} invariant: {violation}")
    return {"label": label, "counts": counts, "invented": len(invented), "read": len(read), "seconds": round(time.time() - started),
            "health": health}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--page", default=os.path.join(folder, "scan.png"))
    parser.add_argument("--chorale", type=int)
    arguments = parser.parse_args()
    chorale, truth = truthNotes(arguments.message, arguments.chorale)
    off = {step: False for step in steps}
    withoutVote = {"vote": False}
    configs = [("legacy: every step off", off), ("all on, one read (no vote)", withoutVote)]
    configs += [(f"all on except {step}", {step: False, "vote": False}) for step in steps[:4]]
    #(the vote is switched off in these, so each row differs from "one read" by that one step only)
    configs += [("all on, with the vote", {})]
    results = [run(arguments.page, truth, label, **switches) for label, switches in configs]
    base = next(r for r in results if r["label"].startswith("all on, one read"))
    lines = [f"ABLATION for {arguments.message!r} (chorale {chorale}, {len(truth)} true notes) on {arguments.page}", "=" * 78,
             f"{'configuration':32s} {'exact':>6s} {'wrongP':>7s} {'missed':>7s} {'invent':>7s} {'read':>5s} {'vs base':>8s} {'secs':>5s}",
             "-" * 78]
    for r in results:
        c = r["counts"]
        lines.append(f"{r['label']:32s} {c['exact']:6d} {c['wrong pitch']:7d} {c['missed']:7d} {r['invented']:7d} {r['read']:5d} "
                     f"{c['exact'] - base['counts']['exact']:+8d} {r['seconds']:5d}")
    lines += ["", "WHAT EACH STEP DID (exact notes with the step on, minus with it off)", "-" * 78]
    for r in results:
        if r["label"].startswith("all on except"):
            step = r["label"].split()[-1]
            gain = base["counts"]["exact"] - r["counts"]["exact"]
            verdict = "HELPS" if gain > 0 else "HURTS (switch it off or fix it)" if gain < 0 else "no effect on this page"
            lines.append(f"{step:10s} {gain:+4d} exact notes: {verdict}")
    final = next(r for r in results if r["label"].endswith("with the vote"))
    lines.append(f"{'vote':10s} {final['counts']['exact'] - base['counts']['exact']:+4d} exact notes vs one read")
    lines += ["", "HEALTH (crashes and broken invariants)", "-" * 78]
    problems = [(r["label"], line) for r in results for line in r["health"]]
    for label, line in problems:
        lines.append(f"[{label}] {line}")
    if not problems:
        lines.append("none")
    text = "\n".join(lines) + "\n"
    print(text)
    with open(os.path.join(os.path.dirname(arguments.page), "ablation.txt"), "w", encoding="utf-8") as file:
        file.write(text)


if __name__ == "__main__":
    main()
