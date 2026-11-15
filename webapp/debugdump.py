#the server's debugging suite. after every scan it writes a folder, webapp/debug/scan-<time>/ (and a copy in
#webapp/debug/latest/ that is always the newest), holding:
#   scan.png        the cleaned page the reader was given
#   readBack.mid    the midi made from the notes the reader found
#   readBack.png    that midi drawn again by the renderer
#   sideBySide.png  scan.png on the left, readBack.png on the right
#   dump.txt        note by note: what should have been there, what was read, and why they differ
#   dump.json       the same facts as data, plus everything the reader saw bar by bar
#the work runs on a background thread, so the phone is not kept waiting
import copy
import json
import os
import shutil
import threading
import time
import traceback

import cv2
import numpy

debugFolder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
keepScans = 15
letters = "CDEFGAB"
pitchClassNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
lock = threading.Lock()


def pitchName(pitch):
    return pitchClassNames[pitch % 12] + str(pitch // 12 - 1)


def names(pitches):
    return " ".join(pitchName(pitch) for pitch in sorted(pitches)) or "nothing"


def stepName(step):
    octave, letter = divmod(step, 7)
    return letters[letter] + str(octave)


def stepPitch(step):
    octave, letter = divmod(step, 7)
    return 12 * (octave + 1) + (0, 2, 4, 5, 7, 9, 11)[letter]


def saveScan(pagePath, notes, ticksPerBeat, decoded, trace, warnings, readError):
    #copies of everything are taken now, the rest happens in the background
    job = (pagePath, list(notes), ticksPerBeat, json.loads(json.dumps(decoded, default=str)), copy.deepcopy(trace),
           list(warnings), readError)
    threading.Thread(target=run, args=job, daemon=True).start()


def run(pagePath, notes, ticksPerBeat, decoded, trace, warnings, readError):
    try:
        write(pagePath, notes, ticksPerBeat, decoded, trace, warnings, readError)
    except Exception:
        print("debug dump failed:\n" + traceback.format_exc(), flush=True)


def render(midiPath, pngPath):
    #the renderer's own picture of the read-back midi. verovio is not thread safe, so this shares the encoder's lock
    import renderer
    import verovio
    from importlib.resources import files
    from chorales import drawLock

    with drawLock:
        verovio.setDefaultResourcePath(str(files("verovio") / "data"))
        renderer.midiToScore(midiPath, pngPath, width=2000, zoom=1.0)


def sideBySide(leftPath, rightPath, outputPath):
    pictures = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in (leftPath, rightPath)]
    if any(picture is None for picture in pictures):
        return False
    #the two are shown at the same height, so a bar in one lines up roughly with the same bar in the other
    height = max(picture.shape[0] for picture in pictures)
    resized = []
    for picture in pictures:
        scale = height / picture.shape[0]
        resized.append(cv2.resize(picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    divider = numpy.full((height, 10), 120, numpy.uint8)
    cv2.imwrite(outputPath, numpy.hstack([resized[0], divider, resized[1]]))
    return True


def write(pagePath, notes, ticksPerBeat, decoded, trace, warnings, readError):
    from reader import notesToMidi

    with lock:
        os.makedirs(debugFolder, exist_ok=True)
        folder = os.path.join(debugFolder, time.strftime("scan-%Y%m%d-%H%M%S"))
        os.makedirs(folder, exist_ok=True)
        scanPath, midiPath = os.path.join(folder, "scan.png"), os.path.join(folder, "readBack.mid")
        renderPath, bothPath = os.path.join(folder, "readBack.png"), os.path.join(folder, "sideBySide.png")
        shutil.copyfile(pagePath, scanPath)
        renderError = None
        try:
            with open(midiPath, "wb") as file:
                file.write(notesToMidi(notes, ticksPerBeat))
            render(midiPath, renderPath)
            sideBySide(scanPath, renderPath, bothPath)
        except Exception as error:
            renderError = f"{type(error).__name__}: {error}"
        dump = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "readError": readError, "warnings": warnings,
                "renderError": renderError, "decoded": decoded, "trace": trace,
                "notes": [{"pitch": n.pitch, "name": pitchName(n.pitch), "startBeat": n.start / ticksPerBeat,
                           "endBeat": n.end / ticksPerBeat, "confidence": n.confidence} for n in notes]}
        with open(os.path.join(folder, "dump.json"), "w", encoding="utf-8") as file:
            json.dump(dump, file, indent=1)
        with open(os.path.join(folder, "dump.txt"), "w", encoding="utf-8") as file:
            file.write(explain(dump))
        latest = os.path.join(debugFolder, "latest")
        shutil.rmtree(latest, ignore_errors=True)
        shutil.copytree(folder, latest)
        for old in sorted(name for name in os.listdir(debugFolder) if name.startswith("scan-"))[:-keepScans]:
            shutil.rmtree(os.path.join(debugFolder, old), ignore_errors=True)
        print("debug dump written to " + folder, flush=True)


def barRecords(trace):
    return {bar["bar"]: bar for system in trace.get("systems", []) for bar in system["bars"]}


def beatX(bar, beat):
    #where a beat of the bar sits across the page: from the symbols the reader did place, else in proportion
    if bar is None:
        return None
    used = sorted({(item["onset"], item["x"]) for item in bar["items"] if item["used"]})
    for onset, x in used:
        if abs(onset - beat) < 0.01:
            return x
    before = [(onset, x) for onset, x in used if onset < beat]
    after = [(onset, x) for onset, x in used if onset > beat]
    if before and after:
        (o1, x1), (o2, x2) = before[-1], after[0]
        return x1 + (x2 - x1) * (beat - o1) / (o2 - o1)
    return bar["left"] + (bar["right"] - bar["left"]) * (0.1 + 0.8 * beat / 4)


def onBar(symbol, bar):
    #inside this bar's own part of the page: across by its barlines, down by its system
    return bar["left"] < symbol["x"] < bar["right"] and bar["top"] <= symbol["y"] <= bar["bottom"]


def whyMissing(pitch, bar, beat, trace):
    #the decoder expected this pitch at this beat and the scan did not give it: look at what the reader had there
    if bar is None:
        return ["no bar was read at this point of the page (the whole bar is missing: no barlines, or no symbols inside them)"]
    spacing = trace.get("spacing", 30)
    x = beatX(bar, beat)
    lines = []
    for item in bar["items"]:
        if abs(item["x"] - x) > 1.0 * spacing:
            continue
        if item["kind"] == "rest":
            lines.append(f"a {item['glyph']} (score {item['score']}) was read at x={item['x']}, {item['beats']:g} beats, "
                         f"{'used' if item['used'] else 'not used'}: a rest here would hide a note")
        else:
            read = stepName(item["step"])
            close = abs(stepPitch(item["step"]) - pitch) <= 1
            lines.append(
                f"a {item['glyph']} (score {item['score']}, grid confidence {item['confidence']}, stem {item['stem']}) was read at "
                f"x={item['x']} as {read}, {'the right line or space' if close else 'a different line or space'}, "
                f"{'and used as ' + item['voice'] + ' voice' if item['used'] else 'but not used by the bar: the voices were already full, or it scored lower than the choice made'}")
    nearby = sorted((m for m in trace.get("nearMisses", []) if onBar(m, bar) and abs(m["x"] - x) <= 1.0 * spacing),
                    key=lambda m: -m["score"])[:4]
    for miss in nearby:
        if True:
            lines.append(f"a near miss: {miss['name']} scored {miss['score']} but needs {miss['need']} (at x={miss['x']}, y={miss['y']})")
    for lost in [m for m in trace.get("suppressed", []) if onBar(m, bar) and abs(m["x"] - x) <= 1.0 * spacing][:4]:
        if True:
            lines.append(f"{lost['name']} ({lost['score']}) was beaten on the same spot by {lost['by']} ({lost['byScore']})")
    if not lines:
        lines.append(f"nothing at all matched near x={round(x)}: the symbol is missing from the picture or too blurred for any template")
    return lines


def whyExtra(pitch, bar, beat, trace):
    #the scan gave a pitch the decoder did not expect: which symbol was it, and was it a misreading
    if bar is None:
        return ["no bar record"]
    spacing = trace.get("spacing", 30)
    x = beatX(bar, beat)
    lines = []
    for item in bar["items"]:
        if item["kind"] != "head" or abs(item["x"] - x) > 1.0 * spacing or not item["used"]:
            continue
        if abs(stepPitch(item["step"]) - pitch) <= 1:
            lines.append(f"read from a {item['glyph']} at x={item['x']}, y={item['y']} (score {item['score']}, grid confidence "
                         f"{item['confidence']}, stem {item['stem']}, ledger support {item['ledgerSupport']}) as {stepName(item['step'])}; "
                         f"if this is the wrong pitch the head sat between a line and a space or the staff lines are a little off there")
    return lines or ["no single symbol could be traced to it (it may come from a held note of an earlier beat)"]


def stepHealth(trace):
    #which of the reader's improvement steps ran, crashed or broke one of its own invariants, with what each measured
    lines = []
    steps = trace.get("steps", {})
    for step in ("curves", "keyVote", "columns", "restGuard", "vote", "dots", "pickup", "nearLines", "shortBars", "deblur"):
        entry = steps.get(step)
        if entry is None:
            lines.append(f"{step:10s} did not run")
            continue
        status = "CRASHED (fell back to the old code)" if entry["crashed"] else "ok" if not entry["violations"] else "BROKE AN INVARIANT"
        lines.append(f"{step:10s} {status}: new code ran {entry['new']}x, old code ran {entry['legacy']}x")
        probes = entry["probes"]
        if step == "curves":
            staffs = probes.get("staff", [])
            if staffs:
                kept = sum(1 for p in staffs if p["kept"])
                curved = [p["offGridCurved"] for p in staffs if p["offGridCurved"] is not None]
                level = [p["offGridLevel"] for p in staffs if p["offGridLevel"] is not None]
                lines.append(f"           {kept} of {len(staffs)} staves kept their curve; heads sit {sum(curved) / max(1, len(curved)):.3f} "
                             f"off the line/space grid with curves vs {sum(level) / max(1, len(level)):.3f} level (lower is better); "
                             f"largest drift {max(p['maxDriftPx'] for p in staffs)} px")
            for item in probes.get("stepsChanged", []):
                lines.append(f"           the curves moved {item['changed']} of {item['heads']} heads by a line or space")
        elif step == "keyVote":
            for item in probes.get("systems", []):
                lines.append(f"           key read in each system {item['reads']}, winner {item['winner']}")
            for item in probes.get("oldWay", []):
                lines.append(f"           the old way (first system only) read {item['read']}, used {item['used']}")
        elif step == "columns":
            adopted = probes.get("adopted", [])
            lines.append(f"           the across-staves columns were adopted in {len(adopted)} bar(s): "
                         + ", ".join(f"bar {a['bar']}" for a in adopted[:12]))
        elif step == "restGuard":
            dropped = probes.get("dropped", [])
            lines.append(f"           {len(dropped)} rest(s) dropped for lying on strong noteheads: "
                         + ", ".join(f"{d['glyph']} at x={d['x']}" for d in dropped[:8]))
        elif step == "vote":
            for item in probes.get("agreement", []):
                lines.append(f"           read at size {item['withRead']} shares {item['share']:.0%} of its notes with the normal read "
                             f"({item['notes'][0]} vs {item['notes'][1]} notes)")
            for item in probes.get("skipped", []):
                lines.append(f"           size {item['size']} skipped: {item['error']}")
        for violation in entry["violations"][:6]:
            lines.append(f"           ! {violation}")
        if entry["error"]:
            lines.append("           ! " + entry["error"].strip().replace("\n", "\n             "))
    return lines


def explain(dump):
    trace, decoded = dump["trace"], dump["decoded"]
    bars = barRecords(trace)
    out = []
    add = out.append
    add(f"SCAN DUMP {dump['time']}")
    add("=" * 60)
    add(f"reader: {'FAILED: ' + dump['readError'] if dump['readError'] else 'read the page'}; {len(dump['notes'])} notes; "
        f"{len(trace.get('staves', []))} staves; staff spacing {trace.get('spacing')} px; {len(bars)} bars read; image {trace.get('size')}")
    add(f"decode: text={decoded.get('text')!r} chorale={decoded.get('chorale')} reads={decoded.get('reads')} "
        f"chords corrected={decoded.get('fixed')} of {decoded.get('total')}; flags={decoded.get('flagCounts')}")
    if dump["renderError"]:
        add(f"RENDER of the read-back midi failed: {dump['renderError']}")
    add("")
    short = [bar for bar in bars.values() if bar["problems"]]
    add("STEP HEALTH (the reader's improvements)")
    add("-" * 60)
    for line in stepHealth(trace):
        add(line)
    add("")
    add("BARS THE READER COULD NOT FILL")
    add("-" * 60)
    for bar in short:
        add(f"bar {bar['bar']}: " + "; ".join(bar["problems"]))
    if not short:
        add("none: every bar adds up to four beats in every voice")
    add("")
    add("NOTE BY NOTE: beats where the decoded chord and the read chord differ")
    add("-" * 60)
    details = decoded.get("beatsDetail") or []
    bad = 0
    for detail in details:
        expected, read = set(detail["decoded"]), set(detail["read"])
        if expected == read:
            continue
        bad += 1
        bar = bars.get(detail["scanBar"])
        beat = detail["scanBeatInBar"]
        add(f"bar {detail['scanBar']} beat {beat + 1:g} (tune beat {detail['beat']}, phrase {detail['phrase'] + 1}"
            f"{'' if detail['phraseResolved'] else ', PHRASE DID NOT DECODE'}): "
            f"decoder wants {names(expected)}; the scan gave {names(read)}; tune soprano {pitchName(detail['soprano'])}")
        add(f"    votes: " + ", ".join(f"{pitchName(int(p))}={v}" for p, v in sorted(detail["votes"].items(), key=lambda i: int(i[0]))))
        for pitch in sorted(expected - read):
            add(f"  MISSING {pitchName(pitch)}:")
            for line in whyMissing(pitch, bar, beat, trace):
                add("      - " + line)
        for pitch in sorted(read - expected):
            add(f"  UNEXPECTED {pitchName(pitch)}:")
            for line in whyExtra(pitch, bar, beat, trace):
                add("      - " + line)
    if not details:
        add("no per-beat comparison: the chorale could not be identified, so there is no expected chord to compare with")
        add(f"flags: {[flag['detail'] for flag in decoded.get('flags', [])][:5]}")
    elif not bad:
        add("none: every beat read matches the decoded chord")
    add("")
    add("BAR DETAIL: every symbol the reader placed in a bar that had a problem")
    add("-" * 60)
    for bar in short:
        add(f"bar {bar['bar']} (x {bar['left']} to {bar['right']}), problems: {'; '.join(bar['problems'])}")
        for item in sorted(bar["items"], key=lambda i: (i["staff"], i["x"])):
            state = f"{item['voice']} voice at beat {item['onset']:g}" if item["used"] else "NOT USED"
            what = f"step {stepName(item['step'])}" if item["step"] is not None else "rest"
            add(f"    {item['staff']:6s} x={item['x']:7} {item['glyph']:13s} score {item['score']} beats {item['beats']:g} {what} -> {state}")
        for other in bar["otherGlyphs"]:
            add(f"    other  x={other['x']:7} {other['name']} score {other['score']}")
        for miss in trace.get("nearMisses", []):
            if onBar(miss, bar) and miss["name"] in ("blackHead", "halfHead", "wholeHead", "quarterRest", "blockRest"):
                add(f"    near miss: {miss['name']} {miss['score']} (needs {miss['need']}) at x={miss['x']}, y={miss['y']}")
    if not short:
        add("none")
    add("")
    add("SYMBOLS OUTSIDE ANY BAR")
    add("-" * 60)
    for orphan in trace.get("orphans", []):
        add(f"system {orphan['system'] + 1}: {orphan['name']} {orphan['score']} at x={orphan['x']}, y={orphan['y']}: {orphan['why']}")
    if not trace.get("orphans"):
        add("none")
    add("")
    add("WARNINGS FROM THE READER")
    add("-" * 60)
    for warning in dump["warnings"]:
        add(warning)
    if not dump["warnings"]:
        add("none")
    return "\n".join(out) + "\n"
