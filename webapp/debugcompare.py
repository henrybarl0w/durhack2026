#compares a scan's dump with what the page SHOULD say, note by note. the truth comes from encoding the message again
#(the same chords the encoder drew), so every missed, misplaced and invented note can be listed with the reason the
#reader's own trace gives for it.
#   python debugcompare.py "Hello world"                 newest dump, chorale chosen as the encoder would
#   python debugcompare.py "Hello world" --chorale 241 --dump debug/scan-20261114-234248
#writes truth_compare.txt and truth_compare.json next to the dump
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chorale_engine
import chorales
from debugdump import beatX, barRecords, pitchName, stepName, stepPitch

voiceNames = ("soprano", "alto", "tenor", "bass")
whites = (0, 2, 4, 5, 7, 9, 11)


def diatonic(pitch):
    #the line-or-space number of a pitch, accidentals counted as the white key below
    octave, semitone = divmod(pitch, 12)
    return octave * 7 + max(i for i, white in enumerate(whites) if white <= semitone)


def truthNotes(message, chorale=None):
    number = chorales.textToInt(message)
    if chorale is None:
        chorale = min((beats, c) for c, (cap, beats) in chorales.loadCapacities().items() if number < cap)[1]
    tune = chorales.loadTune(chorale)
    chords = chorale_engine.encode(chorale, number)
    notes = [(event.midi, float(event.offset), float(event.offset + event.length), 0)
             for event in tune.soprano if event.midi is not None]
    for beat, chord in zip(tune.beats, chords):
        for voice in (1, 2, 3):
            notes.append((chord[voice], float(beat.offset), float(beat.offset + beat.length), voice))
    zero = min(note[1] for note in notes)
    return chorale, [(p, s - zero, e - zero, v) for p, s, e, v in notes]


def match(truth, read, tolerance=0.01):
    #truth and read notes paired in three passes: exact, same time with another pitch, same pitch at another time
    truth = sorted(truth, key=lambda n: (n[1], -n[0]))
    read = [dict(pitch=n["pitch"], start=n["startBeat"], end=n["endBeat"], taken=False) for n in read]
    start = min(n["start"] for n in read)
    for n in read:
        n["start"] -= start
        n["end"] -= start
    results = [None] * len(truth)

    def candidates(index, test):
        return [n for n in read if not n["taken"] and test(truth[index], n)]

    for index, (pitch, begin, end, voice) in enumerate(truth):
        found = candidates(index, lambda t, n: n["pitch"] == t[0] and abs(n["start"] - t[1]) < tolerance)
        if found:
            n = found[0]
            n["taken"] = True
            results[index] = ("exact" if abs(n["end"] - end) < 0.05 else "wrong length", n)
    for index, (pitch, begin, end, voice) in enumerate(truth):
        if results[index]:
            continue
        found = candidates(index, lambda t, n: abs(n["start"] - t[1]) < tolerance)
        if found:
            n = min(found, key=lambda n: abs(n["pitch"] - pitch))
            n["taken"] = True
            results[index] = ("wrong pitch", n)
    for index, (pitch, begin, end, voice) in enumerate(truth):
        if results[index]:
            continue
        found = candidates(index, lambda t, n: n["pitch"] == t[0] and abs(n["start"] - t[1]) < 0.75)
        if found:
            n = min(found, key=lambda n: abs(n["start"] - begin))
            n["taken"] = True
            results[index] = ("wrong time", n)
    return truth, results, [n for n in read if not n["taken"]]


def cause(pitch, begin, dump, bars):
    #what the reader's trace says happened where this note should be: a short reason code and a sentence
    trace = dump["trace"]
    spacing = trace.get("spacing", 30)
    bar = bars.get(int(begin // 4) + 1)
    if bar is None:
        return "no bar", "the reader has no bar here"
    beat = begin % 4
    x = beatX(bar, beat)
    near = [item for item in bar["items"] if abs(item["x"] - x) <= 1.0 * spacing]
    heads = [item for item in near if item["kind"] == "head"]
    rests = [item for item in near if item["kind"] == "rest" and item["used"]]
    right = [item for item in heads if abs(stepPitch(item["step"]) - pitch) <= 1]
    if right and any(item["used"] for item in right):
        return "wrong place", f"the right head was read and used, but at another time or length ({right[0]['glyph']} score {right[0]['score']})"
    if right:
        return "found, unused", f"the right head was found (score {right[0]['score']}, stem {right[0]['stem']}) but the bar's voice choice did not use it"
    if rests:
        return "hidden by rest", f"a {rests[0]['glyph']} (score {rests[0]['score']}) was used here"
    if heads:
        step = min(heads, key=lambda item: abs(diatonic(stepPitch(item['step'])) - diatonic(pitch)))
        delta = diatonic(stepPitch(step["step"])) - diatonic(pitch)
        return "head at wrong height", (f"a head was read here but {delta:+d} line/space(s) from the truth "
                                         f"({stepName(step['step'])}, grid confidence {step['confidence']})")
    misses = [m for m in trace.get("nearMisses", []) if abs(m["x"] - x) <= 1.0 * spacing and bar["top"] <= m["y"] <= bar["bottom"]
              and m["name"] in ("blackHead", "halfHead", "wholeHead")]
    if misses:
        best = max(misses, key=lambda m: m["score"])
        return "weak match", f"nothing passed, the best was a {best['name']} at {best['score']} (needs {best['need']})"
    return "nothing matched", f"no template matched near x={round(x)}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("message")
    parser.add_argument("--chorale", type=int)
    parser.add_argument("--dump", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug", "latest"))
    arguments = parser.parse_args()
    dump = json.load(open(os.path.join(arguments.dump, "dump.json"), encoding="utf-8"))
    chorale, truth = truthNotes(arguments.message, arguments.chorale)
    truth, results, invented = match(truth, dump["notes"])
    bars = barRecords(dump["trace"])
    lines = [f"TRUTH vs SCAN for {arguments.message!r}: chorale {chorale}, {len(truth)} true notes, {len(dump['notes'])} read",
             "=" * 70]
    tally = collections.Counter()
    why = collections.Counter()
    rows = []
    for (pitch, begin, end, voice), result in zip(truth, results):
        kind = result[0] if result else "missed"
        tally[kind] += 1
        if kind == "exact":
            continue
        reason, sentence = cause(pitch, begin, dump, bars)
        why[(kind, reason)] += 1
        got = ""
        if result:
            n = result[1]
            got = f"; read {pitchName(n['pitch'])} at beat {n['start'] % 4 + 1:g} for {n['end'] - n['start']:g}"
        rows.append({"bar": int(begin // 4) + 1, "beat": begin % 4 + 1, "voice": voiceNames[voice], "truth": pitchName(pitch),
                     "kind": kind, "reason": reason, "detail": sentence + got})
        lines.append(f"bar {rows[-1]['bar']:2d} beat {rows[-1]['beat']:g} {voiceNames[voice]:7s} should be {pitchName(pitch):4s} "
                     f"{kind.upper():11s} [{reason}] {sentence}{got}")
    lines += ["", "INVENTED: read notes that match nothing in the truth", "-" * 70]
    for n in sorted(invented, key=lambda n: n["start"]):
        bar = bars.get(int(n["start"] // 4) + 1)
        lines.append(f"bar {int(n['start'] // 4) + 1:2d} beat {n['start'] % 4 + 1:g} read {pitchName(n['pitch'])} for {n['end'] - n['start']:g}")
    lines += ["", "SUMMARY", "-" * 70]
    for kind, count in tally.most_common():
        lines.append(f"{kind:12s} {count:4d}  ({100 * count / len(truth):.0f}%)")
    lines.append(f"invented     {len(invented):4d}")
    lines += ["", "WHY (outcome, reason from the reader's own trace)", "-" * 70]
    for (kind, reason), count in why.most_common():
        lines.append(f"{count:4d}  {kind:11s} {reason}")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(arguments.dump, "truth_compare.txt"), "w", encoding="utf-8") as file:
        file.write(text)
    with open(os.path.join(arguments.dump, "truth_compare.json"), "w", encoding="utf-8") as file:
        json.dump({"summary": dict(tally), "invented": len(invented), "rows": rows}, file, indent=1)
    print("\n".join(lines[-(len(tally) + len(why) + 12):]))


if __name__ == "__main__":
    main()
