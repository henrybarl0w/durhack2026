#text <-> chorale: the web app's encoder and decoder around chorale_engine, renderer and reader
#   text -> integer -> chords hidden in a Bach chorale -> musicxml -> png
#   notes read from a scan -> chords -> (repaired against the engine's legal options) -> integer -> text
import base64
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib.resources import files

import chorale_engine
import renderer
import verovio

capacityFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorale_capacity.json")
#the integer is the message bytes behind one marker byte, so a message with leading zero bytes survives
marker = b"\x01"
#a chorale is only named if this much of its soprano line is found in the scan
minSopranoMatch = 0.5

capacities = None
tunes = {}
#how the chords repaired from a scan become the message: set by app.py to the same decoder the Decode tab uses (the word list,
#with error correction), as wordDecoder(chorale, chords) -> (message, repaired phrase indices). if it is left unset the
#older scheme in this file (the message as one integer) is used
wordDecoder = None
#verovio is not thread safe: one chorale is drawn at a time
drawLock = threading.Lock()


class ChoraleError(Exception):
    pass


def loadCapacities():
    #{chorale id: (capacity, beats)} for every usable chorale, worked out once and kept in a file
    global capacities
    if capacities is not None:
        return capacities
    saved = {}
    if os.path.exists(capacityFile):
        with open(capacityFile) as file:
            saved = json.load(file)
    changed = False
    for chorale in chorale_engine.usable_chorales():
        if str(chorale) not in saved:
            capacity = chorale_engine.capacity_report(chorale)["capacity"]
            saved[str(chorale)] = [str(capacity), len(chorale_engine.load_tune(chorale).beats)]
            changed = True
    if changed:
        with open(capacityFile, "w") as file:
            json.dump(saved, file)
    #capacities are huge integers, kept as text in the file
    capacities = {int(chorale): (int(capacity), beats) for chorale, (capacity, beats) in saved.items()}
    return capacities


def warmUp():
    #decoding a scan compares it with every chorale's tune, and the first time each tune is loaded it is slow:
    #load them all once, when the server starts, so the first scan does not pay for it
    for chorale in loadCapacities():
        loadTune(chorale)


def loadTune(chorale):
    if chorale not in tunes:
        tunes[chorale] = chorale_engine.load_tune(chorale)
    return tunes[chorale]


def textToInt(text):
    return int.from_bytes(marker + text.encode("utf-8"), "big")


def intToText(number):
    data = number.to_bytes((number.bit_length() + 7) // 8, "big")
    if not data.startswith(marker):
        raise ChoraleError("this does not decode to a message")
    try:
        return data[1:].decode("utf-8")
    except UnicodeDecodeError:
        raise ChoraleError("this decodes to garbage, so some notes were misread")


def maxBytes():
    #the most message bytes the biggest chorale holds
    biggest = max(capacity for capacity, _ in loadCapacities().values())
    return (biggest.bit_length() - 1) // 8 - 1


def encodeText(text):
    #returns (png bytes, chorale id); the shortest chorale with room for the message
    if len(text.encode("utf-8")) > 256:
        raise ChoraleError(f"message too long: at most about {maxBytes()} bytes fit in one chorale")
    number = textToInt(text)
    fits = [(beats, chorale) for chorale, (capacity, beats) in loadCapacities().items() if number < capacity]
    if not fits:
        raise ChoraleError(f"message too long: at most about {maxBytes()} bytes fit in one chorale")
    #a few chorales may not draw (verovio rejects the score): fall through to the next shortest that fits
    failed = []
    for _, chorale in sorted(fits):
        try:
            return drawChorale(chorale, number), chorale
        except renderer.RendererError as error:
            failed.append(f"{chorale}: {error}")
    raise ChoraleError("could not draw any chorale that fits (" + "; ".join(failed[:3]) + ")")


def drawChorale(chorale, number):
    #chords -> notes -> a grand staff (treble and bass, two voices on each): half the page of four separate staves,
    #and the layout the reader reads, so what is drawn here is what is scanned back
    chords = chorale_engine.encode(chorale, number)
    tune = loadTune(chorale)
    ticks = 480
    notes = []

    def span(offset, length):
        #snap to a sixteenth-note grid, never shorter than one sixteenth: odd lengths make the renderer write bad xml
        start = round(offset * 4) * (ticks // 4)
        return start, max(start + ticks // 4, round((offset + length) * 4) * (ticks // 4))

    #the soprano keeps the tune's own rhythm; alto, tenor and bass hold one note per beat. voices 0-3 are S A T B
    for event in tune.soprano:
        if event.midi is not None:
            notes.append((event.midi, *span(event.offset, event.length), 0))
    for beat, chord in zip(tune.beats, chords):
        for voice in (1, 2, 3):
            notes.append((chord[voice], *span(beat.offset, beat.length), voice))
    #the engine names flats like B-; the renderer wants Bb, and m for minor
    key = tune.home_key.tonic.replace("-", "b") + ("m" if tune.home_key.mode == "minor" else "")
    #set verovio's font folder again right before drawing, in case the web server's thread lost it
    with drawLock:
        verovio.setDefaultResourcePath(str(files("verovio") / "data"))
        return renderer.notesToScore(notes, ticks, None, "png", 2000, 1.0, (4, 4), key)


def pngToDataUri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def soundingAt(notes, tick):
    #pitches of the notes sounding at a tick; a few ticks of slack for notes starting just late
    return {note.pitch for note in notes if note.start <= tick + 5 and note.end > tick + 5}


_aligned = {}


def alignmentOf(chorale, columns):
    #which of the page's chords (in the order they were read) is each of this chorale's beats: [index into columns, or None for a beat with
    #no chord to line up with]. the soprano is the tune, so the top note of each chord is compared with the tune's soprano at each beat, and
    #the two sequences are aligned the way two spellings of a word are: a chord the reader missed, or one it invented, costs a little where
    #it happens and leaves everything after it in step
    beats = loadTune(chorale).beats
    n, m = len(beats), len(columns)
    if abs(n - m) > 0.25 * max(n, m, 1):
        return [None] * n
    wanted = [beat.soprano_midi % 12 for beat in beats]
    have = [max(chord) % 12 if chord else None for chord in columns]
    gap = 1.2
    cost = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i * gap
    for j in range(1, m + 1):
        cost[0][j] = j * gap
    for i in range(1, n + 1):
        row, above = cost[i], cost[i - 1]
        for j in range(1, m + 1):
            same = 0.0 if have[j - 1] == wanted[i - 1] else 1.0
            row[j] = min(above[j - 1] + same, above[j] + gap, row[j - 1] + gap)
    result = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        same = 0.0 if have[j - 1] == wanted[i - 1] else 1.0
        if abs(cost[i][j] - (cost[i - 1][j - 1] + same)) < 1e-9:
            result[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif abs(cost[i][j] - (cost[i - 1][j] + gap)) < 1e-9:
            i -= 1
        else:
            j -= 1
    return result


def alignedChords(chorale, columns):
    #the page's chords lined up with this chorale's beats (see alignmentOf); a beat with no chord to line up with comes back empty, which the
    #repair treats as missing evidence
    key = (chorale, columns)
    if key not in _aligned:
        if len(_aligned) > 4000:
            _aligned.clear()
        _aligned[key] = [set(columns[j]) if j is not None else set() for j in alignmentOf(chorale, columns)]
    return _aligned[key]


def readChords(chorale, notes, ticksPerBeat):
    #what the scan shows at each of this chorale's beats. a page read by reader.readScore carries its chords in order (notes.columns) and
    #those are lined up with the beats; any other list of notes is looked at by time
    columns = getattr(notes, "columns", None)
    if columns:
        return alignedChords(chorale, columns)
    beats = loadTune(chorale).beats
    first = beats[0].offset
    return [soundingAt(notes, int((beat.offset - first) * ticksPerBeat)) for beat in beats]


def findChorale(reads):
    #the tune is fixed, so the chorale is the one whose soprano line best matches the scans at each beat.
    #pitch names are compared, not exact pitches, so a misread octave still counts, and a pitch seen in any read counts.
    #nothing else is used to choose: no chorale number is given, and the length of the page is not consulted
    best, bestScore = None, 0
    for chorale in loadCapacities():
        beats = loadTune(chorale).beats
        seen = [set() for _ in beats]
        for notes, ticksPerBeat in reads:
            for names, pitches in zip(seen, readChords(chorale, notes, ticksPerBeat)):
                names.update(pitch % 12 for pitch in pitches)
        score = sum(1 for beat, names in zip(beats, seen) if beat.soprano_midi % 12 in names) / len(beats)
        if score > bestScore:
            best, bestScore = chorale, score
    if best is None or bestScore < minSopranoMatch:
        raise ChoraleError(f"could not tell which chorale this is (closest was chorale {best}, "
                           f"{round(100 * bestScore)}% of its soprano notes matched, "
                           f"{sum(len(notes) for notes, _ in reads)} notes read)")
    return best, bestScore


def votesAt(reads, chorale):
    #per beat, how much of the evidence shows each pitch sounding: {pitch: share of the reads that have it}.
    #a note a single read misses or invents counts for little once several reads are combined
    beats = loadTune(chorale).beats
    counts = [{} for _ in beats]
    for notes, ticksPerBeat in reads:
        for tally, pitches in zip(counts, readChords(chorale, notes, ticksPerBeat)):
            for pitch in pitches:
                tally[pitch] = tally.get(pitch, 0) + 1
    return [{pitch: count / len(reads) for pitch, count in tally.items()} for tally in counts]


#how much a chord pitch that was not seen costs against a pitch that was seen but is not in the chord (which costs 1).
#the first is the usual guess; the others are alternatives the repair step tries when that guess does not decode
missWeights = (0.4, 0.15, 0.8, 0.0)
pitchNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def flag(kind, detail, beat=None, bar=None):
    #a problem found in the scan: kept and listed to be checked, never a reason to stop
    return {"kind": kind, "beat": beat, "bar": bar, "detail": detail}


def names(pitches):
    return " ".join(pitchNames[pitch % 12] + str(pitch // 12 - 1) for pitch in sorted(pitches)) or "nothing"


def mismatch(option, votes, missWeight=0.4):
    #how badly a legal chord fits the evidence: a pitch that is seen but not in the chord is a misread (costly, in
    #proportion to how many reads saw it), a chord pitch that is not seen may just be a note the reader missed (cheap)
    cost = sum(share for pitch, share in votes.items() if pitch not in option)
    return cost + missWeight * sum(1 - votes.get(pitch, 0) for pitch in set(option))


def bestPath(chorale, phrase, votes, missWeight=0.4):
    #the legal run of chords through one phrase that fits the evidence best overall. choosing the nearest chord one
    #beat at a time lets one misread chord poison every option after it, so every legal path is weighed together
    previous, beforeBass = chorale_engine.start_state(chorale, phrase)
    states = {(previous, beforeBass): (0.0, [])}
    for beat in range(len(votes)):
        following = {}
        for (before, beforeBefore), (cost, path) in states.items():
            for option in chorale_engine.options(chorale, phrase, beat, before, beforeBefore):
                total = cost + mismatch(option, votes[beat], missWeight)
                key = (tuple(option), before[3] if before is not None else None)
                if key not in following or total < following[key][0]:
                    following[key] = (total, path + [tuple(option)])
        #keep the search small
        states = dict(sorted(following.items(), key=lambda item: item[1][0])[:400])
    return min(states.values(), key=lambda item: item[0])[1]


def decodeReads(reads):
    #reads is a list of (notes, ticksPerBeat), one per completed scan of the same page; more reads, better evidence.
    #it never raises for a bad scan: whatever is wrong comes back as flags, and text is None until it decodes.
    #returns a dict: text, chorale, reads, fixed (chords corrected), total (chords), flags, flagCounts
    report = {"text": None, "chorale": None, "reads": len(reads), "fixed": 0, "total": 0, "flags": []}
    try:
        chorale, match = findChorale(reads)
    except ChoraleError as error:
        report["flags"].append(flag("chorale", str(error)))
        report["flagCounts"] = {"chorale": 1}
        return report
    tune = loadTune(chorale)
    votes = votesAt(reads, chorale)
    report.update(chorale=chorale, total=len(tune.beats), sopranoMatch=round(match, 2))
    flags = report["flags"]
    number, scale, unresolved = 0, 1, 0
    allChords = []
    for phrase, (start, end) in enumerate(tune.phrases):
        #the repair step: try the usual guess, then the alternatives, until the phrase decodes to a valid number
        value, chords = None, None
        for weight in missWeights:
            chords = bestPath(chorale, phrase, votes[start:end], weight)
            try:
                value = chorale_engine.decode_phrase(chorale, phrase, chords)
                break
            except chorale_engine.DecodeError:
                continue
        if value is None:
            unresolved += 1
            flags.append(flag("phrase", f"phrase {phrase + 1} has no legal run of chords that decodes to a valid number",
                              beat=start, bar=int(tune.beats[start].offset // 4) + 1))
        else:
            number += value * scale
        scale *= chorale_engine.phrase_capacity(chorale, phrase)
        allChords.extend(chords)
        for offset, (chord, beatVotes) in enumerate(zip(chords, votes[start:end])):
            beat = start + offset
            bar = int(tune.beats[beat].offset // 4) + 1
            majority = {pitch for pitch, share in beatVotes.items() if share >= 0.5}
            #for the debug dump: where this beat sits in the scan (the scan starts at the tune's first beat), the chord
            #the decoder settled on, what was actually read, and the tune's own soprano note
            scanBeat = float(tune.beats[beat].offset - tune.beats[0].offset)
            report.setdefault("beatsDetail", []).append({
                "beat": beat, "phrase": phrase, "scanBar": int(scanBeat // 4) + 1, "scanBeatInBar": scanBeat % 4,
                "decoded": sorted(chord), "read": sorted(majority), "votes": {str(p): round(v, 2) for p, v in beatVotes.items()},
                "soprano": tune.beats[beat].soprano_midi, "phraseResolved": value is not None,
            })
            if len({pitch % 12 for pitch in majority}) < 3:
                flags.append(flag("voices", f"only {names(majority)} seen where a chord was expected", beat, bar))
            if tune.beats[beat].soprano_midi % 12 not in {pitch % 12 for pitch in beatVotes}:
                flags.append(flag("soprano", f"the tune's soprano note {names([tune.beats[beat].soprano_midi])} was not seen", beat, bar))
            if set(chord) != majority:
                report["fixed"] += 1
                flags.append(flag("corrected", f"saw {names(majority)}, the nearest legal chord is {names(chord)}", beat, bar))
    if wordDecoder is not None:
        #the repaired chords go to the app's own decoder, whose error correction can mend a phrase that would not decode
        try:
            report["text"], repairedPhrases = wordDecoder(chorale, allChords)
            report["repairedPhrases"] = repairedPhrases
            for phrase in repairedPhrases:
                flags.append(flag("phrase", f"phrase {phrase + 1} was damaged and was repaired by the error correction"))
        except Exception as error:
            flags.append(flag("text", str(error)))
    elif not unresolved:
        try:
            report["text"] = intToText(number)
        except ChoraleError as error:
            flags.append(flag("text", str(error)))
    counts = {}
    for item in flags:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    report["flagCounts"] = counts
    #the list sent to the page is capped; the counts above are complete
    report["flags"] = flags[:150]
    return report


def decodeNotes(notes, ticksPerBeat):
    return decodeReads([(notes, ticksPerBeat)])
