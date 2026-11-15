#the Scan tab's decoder: notes read from a photo of a page -> which chorale -> chords (repaired against the engine's legal
#options) -> the message, through the same word decoder as the Decipher tab (app.scan_words)
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chorale_engine

capacityFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorale_capacity.json")
#a chorale is only named if this much of its soprano line is found in the scan
minSopranoMatch = 0.5

capacities = None
tunes = {}
#how the chords repaired from a scan become the message: the same decoder the Decipher tab uses (the word list, with the
#checksum), as wordDecoder(chorale, chords) -> (message, repaired phrase indices). app.py sets it; if it is unset when a page
#is decoded (a tool that loads this file on its own), app.py is loaded to get it
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


#chorales scoring this close to the best are all tried: the collection sets some melodies more than once, and those match equally
tieMargin = 0.02


def rankChorales(reads):
    #every chorale close to the best soprano match, as [(chorale, score)]: the best first, then those that tie with it, the one whose
    #length is nearest the page's first. raises ChoraleError if nothing matches well enough
    scores = {}
    for chorale in loadCapacities():
        beats = loadTune(chorale).beats
        seen = [set() for _ in beats]
        for notes, ticksPerBeat in reads:
            for names, pitches in zip(seen, readChords(chorale, notes, ticksPerBeat)):
                names.update(pitch % 12 for pitch in pitches)
        scores[chorale] = sum(1 for beat, names in zip(beats, seen) if beat.soprano_midi % 12 in names) / len(beats)
    best = max(scores, key=lambda chorale: scores[chorale], default=None)
    bestScore = scores.get(best, 0)
    if best is None or bestScore < minSopranoMatch:
        raise ChoraleError(f"could not tell which chorale this is (closest was chorale {best}, "
                           f"{round(100 * bestScore)}% of its soprano notes matched, "
                           f"{sum(len(notes) for notes, _ in reads)} notes read)")
    chords = max((len(getattr(notes, "columns", ()) or ()) for notes, _ in reads), default=0)
    close = [chorale for chorale, score in scores.items() if score >= bestScore - tieMargin]
    close.sort(key=lambda chorale: (-round(scores[chorale], 6), abs(len(loadTune(chorale).beats) - chords) if chords else 0))
    return [(chorale, scores[chorale]) for chorale in close]


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


#what reading a seen note as a suspension costs (a misread costs 1, a missed note missWeight)
suspensionCost = 0.5


def mismatch(option, votes, missWeight=0.4, before=None):
    #how badly a legal chord fits the evidence: a pitch that is seen but not in the chord is a misread (costly, in
    #proportion to how many reads saw it), a chord pitch that is not seen may just be a note the reader missed (cheap).
    #before: the chord before, when the style writes suspensions. then a seen pitch held over from that chord, a step
    #above a pitch of this one, is that pitch's suspension: on the beat the page shows the held note, and the chord's
    #own note only comes half a beat later (chorale_engine.ornaments.suspensions), so it counts as the chord's note
    #reading a note as suspended costs a little, so a chord the page shows exactly is always preferred
    seen, cost = dict(votes), 0.0
    if before is not None:
        for pitch in [p for p in seen if p not in option and p in before]:
            for step in (1, 2):
                if pitch - step in option and pitch - step not in seen:
                    seen[pitch - step] = seen.pop(pitch)
                    cost += suspensionCost
                    break
    cost += sum(share for pitch, share in seen.items() if pitch not in option)
    return cost + missWeight * sum(1 - seen.get(pitch, 0) for pitch in set(option))


#how many equally good runs of chords are kept per phrase, and how many combinations of them are offered to the checksum
tiedPaths = 4
tiedCombinations = 16


def bestPaths(chorale, phrase, votes, missWeight=0.4, weights=None, suspended=False):
    #the legal runs of chords through one phrase that fit the evidence best, every one that fits exactly as well (up to tiedPaths):
    #[(cost, chords)]. choosing the nearest chord one beat at a time lets one misread chord poison every option after it, so every legal
    #path is weighed together. weights: the style's weights (its legal chords differ); None is the default style. suspended: the style
    #writes suspensions (see mismatch). a set of pitches cannot show which voice doubles a note, and a held note may be a suspension or a
    #plain chord note, so two runs can fit a page equally: the message's checksum tells them apart (decodeTies)
    weights = weights or chorale_engine.default_weights
    previous, beforeBass = chorale_engine.start_state(chorale, phrase, weights)
    states = {(previous, beforeBass): (0.0, [[]])}
    for beat in range(len(votes)):
        following = {}
        for (before, beforeBefore), (cost, paths) in states.items():
            for option in chorale_engine.options(chorale, phrase, beat, before, beforeBefore, weights):
                total = cost + mismatch(option, votes[beat], missWeight, before if suspended and before is not None else None)
                key = (tuple(option), before[3] if before is not None else None)
                extended = [path + [tuple(option)] for path in paths]
                if key not in following or total < following[key][0] - 1e-9:
                    following[key] = (total, extended[:tiedPaths])
                elif abs(total - following[key][0]) <= 1e-9:
                    following[key] = (total, (following[key][1] + extended)[:tiedPaths])
        #keep the search small
        states = dict(sorted(following.items(), key=lambda item: item[1][0])[:400])
    best = min(cost for cost, _ in states.values())
    tied = [(cost, path) for cost, paths in sorted(states.values(), key=lambda item: item[0]) if cost <= best + 1e-9 for path in paths]
    return tied[:tiedPaths]


def decodeReads(reads):
    #reads is a list of (notes, ticksPerBeat), one per completed scan of the same page; more reads, better evidence.
    #it never raises for a bad scan: whatever is wrong comes back as flags, and text is None until it decodes.
    #returns a dict: text, chorale, reads, fixed (chords corrected), total (chords), flags, flagCounts
    report = {"text": None, "chorale": None, "reads": len(reads), "fixed": 0, "total": 0, "flags": []}
    try:
        ranked = rankChorales(reads)
    except ChoraleError as error:
        report["flags"].append(flag("chorale", str(error)))
        report["flagCounts"] = {"chorale": 1}
        return report
    first = None
    for chorale, match in ranked:
        result = decodeChorale(dict(report, flags=[]), reads, chorale, match)
        if result["text"] is not None:
            return result
        first = first or result
    return first


def decodeChorale(report, reads, chorale, match):
    #decodeReads once the chorale is chosen
    tune = loadTune(chorale)
    votes = votesAt(reads, chorale)
    report.update(chorale=chorale, total=len(tune.beats), sopranoMatch=round(match, 2))
    #each writing style has its own legal chords, and the page does not say which style wrote it: the repair is run in every
    #style, and the one whose message decodes wins (else the one whose chords fit the scan best)
    from chorale_engine.styles import STYLES, style_weights
    tried = []
    for style in STYLES:
        try:
            tried.append(decodeInStyle(dict(report, flags=[]), chorale, tune, votes, style, style_weights(style)))
        except chorale_engine.ChoraleRejected:
            continue  # this chorale cannot be written in that style, so the page is not in it
    if not tried:
        report["flags"].append(flag("chorale", f"chorale {chorale} cannot carry a message in any style"))
        report["flagCounts"] = {"chorale": 1}
        return report
    decoded = [item for item in tried if item[1]["text"] is not None]
    cost, best = min(decoded or tried, key=lambda item: item[0])
    return best


def decodeInStyle(report, chorale, tune, votes, style, weights):
    #decodeReads for one style: (how badly the repaired chords fit the scan, the report)
    report["style"] = style
    flags = report["flags"]
    cost = 0.0
    allChords = []
    alternatives = []  #per phrase: (first beat, the other runs that fit as well and decode)
    for phrase, (start, end) in enumerate(tune.phrases):
        #the repair step: try the usual guess, then the alternatives, until the phrase decodes to a valid number
        value, chords = None, None
        for weight in missWeights:
            tied = bestPaths(chorale, phrase, votes[start:end], weight, weights, suspended=style == "fancy")
            decodable = []
            for _fit, path in tied:
                try:
                    decodable.append((chorale_engine.decode_phrase(chorale, phrase, path, weights), path))
                except chorale_engine.DecodeError:
                    continue
            chords = tied[0][1]
            if decodable:
                value, chords = decodable[0]
                if len(decodable) > 1:
                    alternatives.append((start, [path for _v, path in decodable[1:]]))
                break
        cost += sum(mismatch(chord, beatVotes, before=prior if style == "fancy" else None)
                    for chord, beatVotes, prior in zip(chords, votes[start:end], [None] + list(chords[:-1])))
        if value is None:
            flags.append(flag("phrase", f"phrase {phrase + 1} has no legal run of chords that decodes to a valid number",
                              beat=start, bar=int(tune.beats[start].offset // 4) + 1))
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
    if wordDecoder is None:
        import app  # noqa: F401  (sets wordDecoder)
    #the repaired chords go to the app's own decoder, whose checksum can mend a phrase that would not decode
    try:
        report["text"], repairedPhrases = decodeTies(chorale, allChords, alternatives)
        report["repairedPhrases"] = repairedPhrases
        for phrase in repairedPhrases:
            flags.append(flag("phrase", f"phrase {phrase + 1} was damaged and was repaired by the error correction"))
    except Exception as error:
        flags.append(flag("text", str(error)))
    counts = {}
    for item in flags:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    report["flagCounts"] = counts
    #the list sent to the page is capped; the counts above are complete
    report["flags"] = flags[:150]
    return cost, report


def decodeTies(chorale, chords, alternatives):
    #wordDecoder on the repaired chords; if that fails and some phrases had other runs that fit the page as well, the combinations of
    #those (up to tiedCombinations) are tried in turn, and the message's checksum picks the one that is right
    import itertools
    try:
        return wordDecoder(chorale, chords)
    except Exception as error:
        firstError = error
    choices = [[None] + paths for _start, paths in alternatives]
    for tried, picks in enumerate(itertools.islice(itertools.product(*choices), 1, tiedCombinations + 1)):
        trial = list(chords)
        for (start, _paths), path in zip(alternatives, picks):
            if path is not None:
                trial[start:start + len(path)] = path
        try:
            return wordDecoder(chorale, trial)
        except Exception:
            continue
    raise firstError


