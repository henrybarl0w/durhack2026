#the server's benchmark: reads a cleaned page, compares it with the chords that SHOULD be on it, and says how accurate the read is, where it
#misses, and whether the misses sit on smudges. every run is added to webapp/debug/bench_history.json, so two runs (before and after a change to the
#reader) can be compared with --compare.
#
#   python debugbench.py --page debug/latest/scan.png --message "word zippy zoom"     truth made by the app's own encoder
#   python debugbench.py --page page.png --musicxml encoded.musicxml --chorale 215      truth from the MusicXML the encoder wrote
#   python debugbench.py --page debug/latest/scan.png --proxy debug/latest             truth = the chords the repair settled on (a stand-in,
#                                                                                       NOT real truth: only for tracking change)
#   python debugbench.py --compare                                                       the last two runs side by side
#
#it writes webapp/debug/bench-<label>/ with
#   bench.png    the page with a box on every chord: green = read right, amber = read wrong but the repair fixed it, red = still wrong,
#                with the notes that were missed (-) or invented (+) written under the box
#   smudge.png   how sharp the page is, as a heat map (blue = crisp, red = smeared) with the wrong chords outlined: smudges show as red
#                patches under the outlines
#   bench.txt    the numbers and every wrong chord
#   bench.json   the same as data
#stripping the debugging later: delete this file.
import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy

here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
sys.path.insert(0, os.path.dirname(here))

from reader import features  # noqa: E402
from reader.vote import readVoted  # noqa: E402

debugFolder = os.path.join(here, "debug")
historyPath = os.path.join(debugFolder, "bench_history.json")
naturals = {0, 2, 4, 5, 7, 9, 11}
pitchNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def name(pitch):
    return pitchNames[pitch % 12] + str(pitch // 12 - 1)


def names(pitches):
    return " ".join(name(p) for p in sorted(pitches)) or "-"


def truthChords(args):
    #([set of pitches per beat], chorale, how the truth was made)
    if args.message:
        import app
        cid, chords = app.encode_indices(app.words_to_indices(args.message), args.style)
        return [set(chord) for chord in chords], cid, f"encoded from the message ({args.style})"
    if args.musicxml:
        import chorale_engine
        if not args.chorale:
            raise SystemExit("--musicxml needs --chorale <number>")
        return [set(chord) for chord in chorale_engine.read_musicxml(args.chorale, args.musicxml)], args.chorale, "read from the MusicXML"
    if args.proxy:
        data = json.load(open(os.path.join(args.proxy, "dump.json"), encoding="utf-8"))["decoded"]
        return [set(b["decoded"]) for b in data["beatsDetail"]], data["chorale"], "STAND-IN: the repaired chords of that dump (not real truth)"
    raise SystemExit("give the truth: --message, --musicxml (with --chorale) or --proxy <dump folder>")


def classify(truth, read):
    #what is wrong with one chord: lost sharps (a note is read a semitone down from what it is), notes read a step or two off, notes absent,
    #and notes invented. returns {kind: count}
    missing, extra = set(truth) - set(read), set(read) - set(truth)
    kinds = {"lostSharp": 0, "offByStep": 0, "absent": 0, "invented": 0}
    for pitch in sorted(missing):
        if pitch - 1 in extra and pitch % 12 not in naturals:
            kinds["lostSharp"] += 1
            extra.discard(pitch - 1)
            continue
        near = [e for e in extra if 0 < abs(e - pitch) <= 2]
        if near:
            kinds["offByStep"] += 1
            extra.discard(near[0])
        else:
            kinds["absent"] += 1
    kinds["invented"] += len(extra)
    return kinds


def sharpness(gray):
    #local sharpness: the variance of the Laplacian over a window. a smudge flattens it, so low values are smears
    lap = cv2.Laplacian(gray.astype(numpy.float32), cv2.CV_32F, ksize=3)
    mean = cv2.blur(lap, (61, 61))
    return numpy.sqrt(numpy.maximum(cv2.blur(lap * lap, (61, 61)) - mean * mean, 0))


def run(args):
    from chorales import decodeReads
    truth, chorale, how = truthChords(args)
    reads = readVoted(args.page, {})
    return analyse(args.page, reads, decodeReads(reads), truth, chorale, how, args.label, args.message, echo=True)


def analyse(page, reads, decoded, truth, chorale, how, label=None, message=None, echo=False):
    #reads and decoded are what the scanner already worked out (nothing is read again); truth is a list of chords, one set per beat
    from chorales import alignmentOf, loadTune, votesAt
    started = time.time()
    firstNotes = reads[0][0]
    votes = votesAt(reads, chorale)
    read = [{p for p, share in v.items() if share >= 0.5} for v in votes]
    repaired = None
    if decoded.get("chorale") == chorale and decoded.get("beatsDetail"):
        repaired = [set(b["decoded"]) for b in decoded["beatsDetail"]]
    beats = loadTune(chorale).beats
    count = len(beats)
    if len(truth) != count:
        raise ValueError(f"the truth has {len(truth)} chords but chorale {chorale} has {count} beats")

    gray = cv2.imread(page, cv2.IMREAD_GRAYSCALE)
    sharp = sharpness(gray)
    scale = getattr(firstNotes, "workScale", 1.0) or 1.0
    spacing = 30.0 / scale
    mapping = alignmentOf(chorale, firstNotes.columns) if firstNotes.columns else [None] * count
    positions = firstNotes.columnPositions
    boxes = []
    for index in range(count):
        j = mapping[index]
        if j is None or j >= len(positions) or positions[j][0] is None:
            boxes.append(None)
            continue
        x, top, bottom = positions[j]
        #the reader's window for a system reaches into the margin above and below it: keep the middle of it, which is the two staves
        margin = 0.17 * (bottom - top)
        boxes.append((x / scale, (top + margin) / scale, (bottom - margin) / scale))

    rows, totals = [], {"lostSharp": 0, "offByStep": 0, "absent": 0, "invented": 0}
    rightRead = rightRepaired = sopranoRight = bassRight = found = extras = notesTotal = 0
    sharpTotal = sharpFound = 0
    smudgeRight, smudgeWrong = [], []
    for index in range(count):
        t, r = truth[index], read[index]
        fixed = repaired[index] if repaired else None
        notesTotal += len(t)
        found += len(t & r)
        extras += len(r - t)
        for pitch in t:
            if pitch % 12 not in naturals:
                sharpTotal += 1
                sharpFound += pitch in r
        sopranoRight += bool(r) and max(r) == max(t)
        bassRight += bool(r) and min(r) == min(t)
        exact = r == t
        rightRead += exact
        fixedRight = fixed == t if fixed is not None else False
        rightRepaired += fixedRight
        local = None
        if boxes[index]:
            x, top, bottom = boxes[index]
            y0, y1 = int(max(top, 0)), int(min(bottom, sharp.shape[0] - 1))
            x0, x1 = int(max(x - 1.5 * spacing, 0)), int(min(x + 1.5 * spacing, sharp.shape[1] - 1))
            if y1 > y0 and x1 > x0:
                local = float(numpy.median(sharp[y0:y1, x0:x1]))
                (smudgeRight if exact else smudgeWrong).append(local)
        if not exact:
            kinds = classify(t, r)
            for kind, number in kinds.items():
                totals[kind] += number
            rows.append({"beat": index, "bar": int(beats[index].offset // 4) + 1, "truth": names(t), "read": names(r),
                         "missing": names(t - r), "extra": names(r - t), "repairedRight": fixedRight, "kinds": kinds,
                         "sharpness": None if local is None else round(local, 1)})
    metrics = {
        "chords": count, "chordsReadRight": rightRead, "chordsRightAfterRepair": rightRepaired if repaired else None,
        "voiceNotes": notesTotal, "notesFound": found, "notesInvented": extras,
        "noteRecall": round(found / notesTotal, 3), "notePrecision": round(found / max(1, found + extras), 3),
        "sopranoRight": sopranoRight, "bassRight": bassRight,
        "sharpNotes": sharpTotal, "sharpNotesFound": sharpFound,
        "misses": totals,
        "sharpnessRightChords": round(float(numpy.mean(smudgeRight)), 1) if smudgeRight else None,
        "sharpnessWrongChords": round(float(numpy.mean(smudgeWrong)), 1) if smudgeWrong else None,
        "words": decoded.get("text"), "wordsRight": None if not message else decoded.get("text") == message,
        "chorale": chorale, "chorale identified": decoded.get("chorale"), "reads": len(reads),
        "chordsNeverPlaced": sum(box is None for box in boxes),
    }

    label = label or time.strftime("%H%M%S")
    folder = os.path.join(debugFolder, "bench-" + label)
    os.makedirs(folder, exist_ok=True)
    draw(page, boxes, spacing, truth, read, repaired, sharp, folder)
    report(metrics, rows, how, folder, round(time.time() - started, 1), echo)
    history = json.load(open(historyPath)) if os.path.exists(historyPath) else []
    history.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "label": label, "page": os.path.basename(page), "truth": how,
                    "switches": dict(features.active), "metrics": metrics})
    json.dump(history, open(historyPath, "w"), indent=1)
    return metrics, folder


def draw(page, boxes, spacing, truth, read, repaired, sharp, folder):
    image = cv2.cvtColor(cv2.imread(page, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap(cv2.normalize(-sharp, None, 0, 255, cv2.NORM_MINMAX).astype(numpy.uint8), cv2.COLORMAP_JET)
    smudge = cv2.addWeighted(image, 0.45, heat, 0.55, 0)
    for index, box in enumerate(boxes):
        if box is None:
            continue
        x, top, bottom = box
        exact = read[index] == truth[index]
        fixed = repaired is not None and repaired[index] == truth[index]
        colour = (0, 160, 0) if exact else ((0, 170, 230) if fixed else (0, 0, 220))
        p1, p2 = (int(x - 0.9 * spacing), int(top)), (int(x + 0.9 * spacing), int(bottom))
        cv2.rectangle(image, p1, p2, colour, 3 if not exact else 2)
        if not exact:
            cv2.rectangle(smudge, p1, p2, (255, 255, 255), 3)
            miss, extra = truth[index] - read[index], read[index] - truth[index]
            text = ("-" + " -".join(name(p) for p in sorted(miss)) if miss else "") + (" +" + " +".join(name(p) for p in sorted(extra)) if extra else "")
            cv2.putText(image, text.strip(), (p1[0] - 10, p2[1] + int(0.8 * spacing)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * spacing / 14, colour, 2)
    cv2.imwrite(os.path.join(folder, "bench.png"), image)
    cv2.imwrite(os.path.join(folder, "smudge.png"), smudge)


def report(metrics, rows, how, folder, seconds, echo=True):
    m = metrics
    lines = [f"BENCHMARK  truth: {how}", "=" * 70,
             f"chords read exactly right    {m['chordsReadRight']:>3} / {m['chords']}  ({100 * m['chordsReadRight'] / m['chords']:.0f}%)",
             "chords right after repair    " + (f"{m['chordsRightAfterRepair']:>3} / {m['chords']}  ({100 * m['chordsRightAfterRepair'] / m['chords']:.0f}%)"
                                               if m["chordsRightAfterRepair"] is not None else "(chorale not identified, so no repair)"),
             f"voice notes found           {m['notesFound']:>3} / {m['voiceNotes']}  (recall {100 * m['noteRecall']:.0f}%, precision {100 * m['notePrecision']:.0f}%)",
             f"soprano right / bass right   {m['sopranoRight']} / {m['bassRight']}  of {m['chords']}",
             f"sharp/flat notes found       {m['sharpNotesFound']} / {m['sharpNotes']}",
             f"how the misses split         {m['misses']}",
             f"sharpness (higher = crisper) right chords {m['sharpnessRightChords']}, wrong chords {m['sharpnessWrongChords']}",
             f"chords the reader never placed on the page: {m['chordsNeverPlaced']}",
             f"words: {m['words']!r}" + ("" if m["wordsRight"] is None else "  (right)" if m["wordsRight"] else "  (WRONG)"),
             f"time {seconds}s", "", "EVERY CHORD READ WRONG (beat, bar: truth | read | missing, extra | repair fixed it?)", "-" * 70]
    for row in rows:
        lines.append(f"beat {row['beat']:>2} bar {row['bar']:>2}: {row['truth']} | {row['read']} | -{row['missing']} +{row['extra']} | "
                     f"{'fixed' if row['repairedRight'] else 'STILL WRONG'} | sharpness {row['sharpness']}")
    text = "\n".join(lines)
    open(os.path.join(folder, "bench.txt"), "w", encoding="utf-8").write(text)
    json.dump({"metrics": metrics, "wrong": rows}, open(os.path.join(folder, "bench.json"), "w"), indent=1)
    if echo:
        print(text)
        print("\nwritten to", folder)


truthFile = os.path.join(here, "truth.json")
keepBenches = 15
_lock = threading.Lock()


def liveTruth(decoded):
    #the answer key for a scan that has just come in. webapp/truth.json says what is on the page being scanned:
    #   {"message": "word zippy zoom", "style": "original"}       (made with the app's own encoder)
    #   {"musicxml": "path/to/encoded.musicxml", "chorale": 215}  (the file the encoder wrote)
    #without that file the repaired chords stand in for the truth: the plot and the smudge map still work, and a change between runs still
    #shows, but the numbers say how much the repair had to change, not how right the read is
    if os.path.exists(truthFile):
        config = json.load(open(truthFile, encoding="utf-8"))
        arguments = argparse.Namespace(message=config.get("message"), musicxml=config.get("musicxml"), chorale=config.get("chorale"),
                                       proxy=None, style=config.get("style", "original"))
        truth, chorale, how = truthChords(arguments)
        return truth, chorale, how, config.get("message")
    if decoded.get("chorale") is None or not decoded.get("beatsDetail"):
        raise ValueError("no truth.json, and the scan did not identify a chorale to stand in for it")
    return ([set(b["decoded"]) for b in decoded["beatsDetail"]], decoded["chorale"],
            "STAND-IN: the repaired chords (no webapp/truth.json to say what is on the page)", None)


def benchScan(pagePath, reads, decoded):
    #called by the scanner after every scan: copies are taken now and the benchmark runs in the background, from the reads already made
    job = (pagePath, list(reads), json.loads(json.dumps(decoded, default=str)))
    threading.Thread(target=_benchRun, args=job, daemon=True).start()


def _benchRun(pagePath, reads, decoded):
    import shutil
    import traceback
    try:
        with _lock:
            truth, chorale, how, message = liveTruth(decoded)
            _, folder = analyse(pagePath, reads, decoded, truth, chorale, how, time.strftime("scan-%Y%m%d-%H%M%S"), message)
            latest = os.path.join(debugFolder, "latest_bench")
            shutil.rmtree(latest, ignore_errors=True)
            shutil.copytree(folder, latest)
            old = sorted(n for n in os.listdir(debugFolder) if n.startswith("bench-scan-"))[:-keepBenches]
            for name in old:
                shutil.rmtree(os.path.join(debugFolder, name), ignore_errors=True)
            print("benchmark written to " + folder, flush=True)
    except Exception:
        print("benchmark failed:", traceback.format_exc(), flush=True)


def compare():
    history = json.load(open(historyPath)) if os.path.exists(historyPath) else []
    if len(history) < 2:
        raise SystemExit("need two runs in bench_history.json to compare")
    a, b = history[-2], history[-1]
    print(f"before: {a['label']} ({a['time']})   after: {b['label']} ({b['time']})")
    for key in ("chordsReadRight", "chordsRightAfterRepair", "notesFound", "notesInvented", "sopranoRight", "bassRight", "sharpNotesFound",
                "chordsNeverPlaced"):
        print(f"  {key:<24} {a['metrics'].get(key)!s:>6} -> {b['metrics'].get(key)!s:<6}")
    for kind in a["metrics"]["misses"]:
        print(f"  miss: {kind:<18} {a['metrics']['misses'][kind]:>6} -> {b['metrics']['misses'][kind]:<6}")
    print("  words:", a["metrics"]["words"], "->", b["metrics"]["words"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--page")
    parser.add_argument("--message")
    parser.add_argument("--musicxml")
    parser.add_argument("--chorale", type=int)
    parser.add_argument("--proxy")
    parser.add_argument("--style", default="original")
    parser.add_argument("--label")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--switch", action="append", default=[], help="e.g. --switch weakSharps=off (to compare a step on and off)")
    arguments = parser.parse_args()
    if arguments.compare:
        compare()
    else:
        if not arguments.page:
            raise SystemExit("--page is needed")
        for item in arguments.switch:
            key, value = item.split("=")
            features.active[key] = value.lower() not in ("off", "0", "false")
        run(arguments)
