import warnings
from collections import namedtuple

import cv2
import numpy

from .glyphs import accidentalNames, findGlyphs, flagNames, furnitureNames, headNames
from .staves import System, eraseLines, findBarlines, findStaves, lightLevel, loadGray

Note = namedtuple("Note", ["pitch", "start", "end", "confidence"])

ticksPerBeat = 480
beatsPerBar = 4
#the image is rescaled so a staff space is this many pixels, every other size is a fraction of a staff space
workingSpacing = 30
letterSemitones = (0, 2, 4, 5, 7, 9, 11)
#diatonic step of the bottom line of each staff, counted as octave * 7 + letter: E4 and G2
bottomLineSteps = {"treble": 4 * 7 + 2, "bass": 2 * 7 + 4}
beatsByGlyph = {
    "wholeHead": 4, "halfHead": 2, "blackHead": 1, "quarterRest": 1, "eighthRest": 0.5, "sixteenthRest": 0.25,
    "flag8Up": 0.5, "flag8Down": 0.5, "flag16Up": 0.25, "flag16Down": 0.25,
}
alterByGlyph = {"sharp": 1, "flat": -1, "natural": 0}
#letters (C is 0) in the order a key signature adds its sharps or its flats
sharpOrder = (3, 0, 4, 1, 5, 2, 6)
flatOrder = (6, 2, 5, 1, 4, 0, 3)
#how far below the middle of the glyph the note it marks sits, in staff spaces: a flat's bowl is at its foot
accidentalDrop = {"sharp": 0.0, "natural": 0.0, "flat": 0.57}
#symbols at the same onset touch or overlap, a notehead pushed aside by a second or a unison included;
#symbols at different onsets always have a gap between them
columnSlack = 0.15

#one symbol with its length in beats, its pitch (None for a rest) and how sure we are of it
Event = namedtuple("Event", ["x", "y", "width", "beats", "pitch", "confidence", "isWholeRest"])


def stepToPitch(step):
    octave, letter = divmod(step, 7)
    return 12 * (octave + 1) + letterSemitones[letter]


def normalise(gray):
    staves, spacing = findStaves(gray)
    factor = workingSpacing / spacing
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    gray = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=interpolation)
    staves, spacing = findStaves(gray)
    return gray, staves, spacing


def lineStrength(gray, y, left, right, spacing):
    #how much darker the darkest row near y is than the rows around it, 0 to 1:
    #a thin line stands out even when blurred, the soft edge of a nearby symbol does not
    top, bottom = max(0, int(round(y - 0.35 * spacing))), min(gray.shape[0], int(round(y + 0.35 * spacing)) + 1)
    left, right = max(0, int(round(left))), min(gray.shape[1], int(round(right)) + 1)
    if bottom - top < 3 or left >= right:
        return 0.0
    darkness = (255.0 - gray[top:bottom, left:right]).mean(axis=1) / 255
    rows = numpy.arange(top, bottom)
    near = darkness[numpy.abs(rows - y) <= 0.12 * spacing]
    return float(near.max() - numpy.median(darkness)) if len(near) else 0.0


def lineBeside(gray, y, match, spacing):
    #how clearly a staff or ledger line at height y sticks out on both sides of the symbol
    edge = match.width / 2
    reach = 0.2 * spacing
    leftSide = lineStrength(gray, y, match.cx - edge - reach, match.cx - edge, spacing)
    rightSide = lineStrength(gray, y, match.cx + edge, match.cx + edge + reach, spacing)
    return min(leftSide, rightSide)


def headStaff(system, match, gray, spacing):
    #a note between the staves belongs to the staff its ledger lines lead back to
    choices = []
    for staffName, staff in (("treble", system.treble), ("bass", system.bass)):
        if staff.lines[0] - spacing / 4 <= match.cy <= staff.lines[4] + spacing / 4:
            return staffName, staff
        above = match.cy < staff.lines[0]
        distance = staff.lines[0] - match.cy if above else match.cy - staff.lines[4]
        edge = staff.lines[0] if above else staff.lines[4]
        direction = -1 if above else 1
        #every ledger line from the staff out to the note has to be there
        ledgerCount = int(distance / spacing + 0.25)
        isValid = all(
            lineBeside(gray, edge + direction * spacing * step, match, spacing) > 0.15
            for step in range(1, ledgerCount + 1)
        )
        #when both staves fit (their ledger lines can line up), the longer chain of ledger lines wins
        choices.append((not isValid, -ledgerCount if isValid else 0, distance, staffName, staff))
    staffName, staff = min(choices)[-2:]
    return staffName, staff


def hasLedgerSupport(match, system, gray, spacing):
    #inside a staff, or just outside it (no ledger line needed), or joined to a staff by ledger lines
    for staff in (system.treble, system.bass):
        above = match.cy < staff.lines[0]
        distance = staff.lines[0] - match.cy if above else match.cy - staff.lines[4]
        if distance <= 0.75 * spacing:
            return True
        edge = staff.lines[0] if above else staff.lines[4]
        direction = -1 if above else 1
        ledgerCount = int(distance / spacing + 0.25)
        if all(
            lineBeside(gray, edge + direction * spacing * step, match, spacing) > 0.15
            for step in range(1, ledgerCount + 1)
        ):
            return True
    return False


def headPlace(match, system, gray, spacing):
    #which staff the head is on, its diatonic step and how well it sits on the grid of lines and spaces
    staffName, staff = headStaff(system, match, gray, spacing)
    halfSteps = (staff.lines[4] - match.cy) / (spacing / 2)
    step = round(halfSteps)
    offGrid = abs(halfSteps - step)
    confidence = round(float(max(0.0, min(1.0, match.score) * (1 - offGrid))), 3)
    return staffName, bottomLineSteps[staffName] + step, confidence


def sameColumn(first, second, spacing):
    #symbols at one onset are stacked on top of each other; the only side by side ones are noteheads
    #pushed aside by a second or a unison, which touch and are at most a step apart in height
    left = max(first.x - first.width / 2, second.x - second.width / 2)
    right = min(first.x + first.width / 2, second.x + second.width / 2)
    if right - left >= 0.5 * min(first.width, second.width):
        return True
    return right - left > -columnSlack * spacing and abs(first.y - second.y) <= 0.6 * spacing


def staffOf(system, y, spacing):
    for staffName, staff in (("treble", system.treble), ("bass", system.bass)):
        if staff.lines[0] - 1.5 * spacing <= y <= staff.lines[4] + 1.5 * spacing:
            return staffName
    return None


def keySignature(glyphs, system, spacing):
    #the sharps or flats between the clef and the time signature of the first system, as (count, alter)
    fours = [glyph for glyph in glyphs if glyph.name == "four"]
    if not fours:
        return 0, 0
    timeLeft = min(glyph.cx - glyph.width / 2 for glyph in fours)
    marks = [
        glyph for glyph in glyphs
        if glyph.name in ("sharp", "flat") and system.barlines[0] < glyph.cx < timeLeft
        and staffOf(system, glyph.cy, spacing) == "treble"
    ]
    if not marks:
        return 0, 0
    return len(marks), alterByGlyph[marks[0].name]


def keyAlters(count, alter):
    order = sharpOrder if alter > 0 else flatOrder
    return {letter: alter for letter in order[:count]}


def headerRight(glyphs, system, spacing, keyCount):
    #clef, key signature and time signature follow each other from the start of the system;
    #only as many sharps or flats as the key has belong to it, a later one is on the first note
    header = sorted(
        (glyph for glyph in glyphs if glyph.name in furnitureNames + accidentalNames),
        key=lambda glyph: glyph.cx - glyph.width / 2,
    )
    end = systemStart = system.barlines[0]
    #the clef and time signature are always near the start; in a photo the key signature between them may be missed
    nearStart = [
        glyph.cx + glyph.width / 2 for glyph in header
        if glyph.name in furnitureNames and systemStart < glyph.cx < systemStart + 15 * spacing
    ]
    end = max(nearStart, default=systemStart)
    used = {"treble": 0, "bass": 0, None: keyCount}
    for glyph in header:
        #the clef may sit a little way in from where the staff starts; after it, the header is unbroken
        allowed = 1.5 * spacing
        if glyph.cx < end:
            continue
        if glyph.cx - glyph.width / 2 > end + allowed:
            break
        if glyph.name in accidentalNames:
            staffName = staffOf(system, glyph.cy, spacing)
            if end == systemStart or used[staffName] >= keyCount:
                continue
            used[staffName] += 1
        end = max(end, glyph.cx + glyph.width / 2)
    return end


def hasStem(ink, match, spacing, direction):
    #an up stem leaves the right side of the head going up, a down stem the left side going down
    side = match.cx + match.width / 2 if direction < 0 else match.cx - match.width / 2
    left, right = int(round(side - 0.3 * spacing)), int(round(side + 0.3 * spacing)) + 1
    near, far = match.cy + direction * 0.4 * spacing, match.cy + direction * 2.5 * spacing
    top, bottom = int(round(min(near, far))), int(round(max(near, far))) + 1
    if top < 0 or bottom > ink.shape[0] or left < 0 or right > ink.shape[1]:
        return False
    return float(ink[top:bottom, left:right].mean(axis=0).max()) > 0.95


def stemJoins(ink, head, flag, spacing, isUp):
    #a flag only belongs to a note if one unbroken stem of ink runs from the notehead to the flag
    stemX = flag.cx - flag.width / 2
    left, right = int(round(stemX - 0.3 * spacing)), int(round(stemX + 0.3 * spacing)) + 1
    if isUp:
        top, bottom = flag.cy - flag.height / 2, head.cy - 0.4 * spacing
    else:
        top, bottom = head.cy + 0.4 * spacing, flag.cy + flag.height / 2
    top, bottom = int(round(top)), int(round(bottom)) + 1
    if top >= bottom or top < 0 or bottom > ink.shape[0] or left < 0 or right > ink.shape[1]:
        return False
    return float(ink[top:bottom, left:right].mean(axis=0).max()) > 0.9


def attachFlags(flags, heads, ink, spacing):
    #a flag hangs on the far end of a stem: an up stem leaves the right side of its head, a down stem the left.
    #returns {head: {"up": beats, "down": beats}}
    attached = {}
    for flag in flags:
        isUp = flag.name.endswith("Up")
        flagLeft = flag.cx - flag.width / 2
        choices = []
        for head in heads:
            if head.name != "blackHead":
                continue
            stemX = head.cx + head.width / 2 - 0.25 * spacing if isUp else head.cx - head.width / 2
            rise = head.cy - flag.cy if isUp else flag.cy - head.cy
            if abs(flagLeft - stemX) < 0.45 * spacing and spacing < rise < 9 * spacing and stemJoins(ink, head, flag, spacing, isUp):
                choices.append((rise, head))
        if choices:
            head = min(choices, key=lambda choice: choice[0])[1]
            attached.setdefault(head, {})["up" if isUp else "down"] = beatsByGlyph[flag.name]
    return attached


def attachAccidentals(accidentals, heads, spacing):
    #an accidental belongs to the nearest head to its right at its own height
    attached = {}
    for accidental in accidentals:
        y = accidental.cy + accidentalDrop[accidental.name] * spacing
        right = [
            head for head in heads
            if 0 < head.cx - accidental.cx < 5 * spacing and abs(head.cy - y) < 0.4 * spacing
        ]
        if right:
            attached[min(right, key=lambda head: head.cx - accidental.cx)] = alterByGlyph[accidental.name]
    return attached


voiceNames = {("treble", "up"): "soprano", ("treble", "down"): "alto", ("bass", "up"): "tenor", ("bass", "down"): "bass"}


def stemDirection(ink, match, flags, spacing):
    #"up", "down" or None (a whole note, or a stem we cannot tell apart)
    if len(flags) == 1:
        return next(iter(flags))
    up, down = hasStem(ink, match, spacing, -1), hasStem(ink, match, spacing, 1)
    if up != down:
        return "up" if up else "down"
    return None


#how much each piece of evidence counts when choosing a bar's notes, in the same units as a match score
stemAgrees = 0.15
stemDisagrees = -0.25
noLedgerSupport = -0.3


def staffColumns(items, spacing):
    #the heads of one staff, grouped into columns of notes that start together
    groups = []
    for item in sorted(items, key=lambda item: item["glyph"].cx):
        event = Event(item["glyph"].cx, item["glyph"].cy, item["glyph"].width, 0, 0, 0, False)
        if groups and any(
            sameColumn(event, Event(other["glyph"].cx, other["glyph"].cy, other["glyph"].width, 0, 0, 0, False), spacing)
            for other in groups[-1]
        ):
            groups[-1].append(item)
        else:
            groups.append([item])
    return groups


def columnChoices(group):
    #every way one column can be read: nothing, one head in either voice, or two heads (higher one upper)
    yield ()
    for item in group:
        yield ((item, "up"),)
        yield ((item, "down"),)
    for first in group:
        for second in group:
            if first is not second and first["glyph"].cy < second["glyph"].cy - 0.25 * first["glyph"].height:
                yield ((first, "up"), (second, "down"))


def evidence(item, direction):
    #how strongly this head, read as a note of this voice, is backed up by what is on the page
    value = item["score"]
    if item["stem"] is not None:
        value += stemAgrees if item["stem"] == direction else stemDisagrees
    if not item["supported"]:
        value += noLedgerSupport
    return value


def chooseStaff(items, spacing):
    #of all the ways to read this staff's heads as two voices, keep the one with the most evidence that fits
    #what is always true: no rests, so when one voice starts a note the other is either starting one too
    #or still holding one, and neither voice runs past the end of the bar.
    #returns (timed, upperBeats, lowerBeats) where timed is [(onset, item, direction)]
    states = {(0.0, 0.0): (0.0, ())}
    for group in staffColumns(items, spacing):
        next = dict(states)
        for (upper, lower), (value, path) in states.items():
            for choice in columnChoices(group):
                if not choice:
                    continue
                clocks = {"up": upper, "down": lower}
                directions = [direction for _, direction in choice]
                if len(choice) == 2 and upper != lower:
                    continue
                if len(choice) == 1:
                    other = "down" if directions[0] == "up" else "up"
                    if clocks[other] <= clocks[directions[0]]:
                        continue
                gained = 0.0
                added = []
                for item, direction in choice:
                    added.append((clocks[direction], item, direction))
                    clocks[direction] += item["beats"]
                    #weighted by the beats the note covers, so two weak blobs cannot outvote one long note
                    gained += evidence(item, direction) * item["beats"]
                if clocks["up"] > beatsPerBar or clocks["down"] > beatsPerBar:
                    continue
                key = (clocks["up"], clocks["down"])
                if key not in next or next[key][0] < value + gained:
                    next[key] = (value + gained, path + tuple(added))
        states = next
    full = (float(beatsPerBar), float(beatsPerBar))
    if full in states:
        value, path = states[full]
        return list(path), beatsPerBar, beatsPerBar
    #no reading fills the bar: keep the one that covers the most beats, then has the most evidence
    (upper, lower), (value, path) = max(states.items(), key=lambda entry: (entry[0][0] + entry[0][1], entry[1][0]))
    return list(path), upper, lower


def chooseBar(placed, spacing):
    #placed: dicts with glyph, staffName, beats, step, confidence, score, stem, supported.
    #returns ([(onset, Event, voiceName)], [problems])
    timed, problems = [], []
    for staffName in ("treble", "bass"):
        path, upper, lower = chooseStaff([item for item in placed if item["staffName"] == staffName], spacing)
        for onset, item, direction in path:
            glyph = item["glyph"]
            event = Event(glyph.cx, glyph.cy, glyph.width, item["beats"], item["step"], item["confidence"], False)
            timed.append((onset, event, voiceNames[(staffName, direction)]))
        for direction, total in (("up", upper), ("down", lower)):
            if total != beatsPerBar:
                problems.append(f"the {voiceNames[(staffName, direction)]} has only {total:g} of {beatsPerBar} beats")
    return timed, problems


def readScore(imagePath):
    gray, staves, spacing = normalise(loadGray(imagePath))
    #thin lines and stems blur when a small image is scaled up, so they are looked for in anything not white
    lineInk = gray < lightLevel
    systems = [
        System(staves[index], staves[index + 1], findBarlines(lineInk, staves[index], staves[index + 1], spacing))
        for index in range(0, len(staves), 2)
    ]
    #in a photo the line joining the staves at the start of a system can be lost: start at the staff lines then
    for index, system in enumerate(systems):
        staffStart = min(system.treble.left, system.bass.left)
        staffEnd = max(system.treble.right, system.bass.right)
        barlines = list(system.barlines)
        if barlines[0] - staffStart > 3 * spacing:
            barlines.insert(0, staffStart)
        #likewise the closing barline when the photo cuts off the end of the score
        if staffEnd - barlines[-1] > 3 * spacing:
            barlines.append(staffEnd)
        systems[index] = system._replace(barlines=barlines)
    glyphs, dots = findGlyphs(eraseLines(gray, staves, spacing), spacing)

    notes = []
    barNumber = 0
    keyCount, keyAlter = 0, 0
    for systemIndex, system in enumerate(systems):
        top = system.treble.lines[0] - 8 * spacing
        bottom = system.bass.lines[4] + 8 * spacing
        if systemIndex > 0:
            top = max(top, (systems[systemIndex - 1].bass.lines[4] + system.treble.lines[0]) / 2)
        if systemIndex + 1 < len(systems):
            bottom = min(bottom, (system.bass.lines[4] + systems[systemIndex + 1].treble.lines[0]) / 2)
        inSystem = [glyph for glyph in glyphs if top <= glyph.cy < bottom]
        if systemIndex == 0:
            keyCount, keyAlter = keySignature(inSystem, system, spacing)
        #clefs, key signature and time signature: nothing up to the end of the last of them is a note
        headerEnd = headerRight(inSystem, system, spacing, keyCount)
        inSystem = [glyph for glyph in inSystem if glyph.cx > headerEnd]
        heads = [glyph for glyph in inSystem if glyph.name in headNames]
        flags = attachFlags([glyph for glyph in inSystem if glyph.name in flagNames], heads, lineInk, spacing)
        accidentals = attachAccidentals([glyph for glyph in inSystem if glyph.name in accidentalNames], heads, spacing)

        for left, right in zip(system.barlines, system.barlines[1:]):
            inBar = sorted((glyph for glyph in inSystem if left < glyph.cx < right), key=lambda glyph: glyph.cx)
            if not inBar:
                continue
            barName = f"bar {barNumber + 1}"
            barStart = barNumber * beatsPerBar
            #a chorale bar: four voices, two per staff, no rests. every head with its staff, length, step and the evidence for it; nothing is thrown away here,
            #the best-fit choice weighs it all. accidentals are applied per voice once voices are known
            placed = []
            for glyph in inBar:
                if glyph.name in headNames:
                    staffName, step, confidence = headPlace(glyph, system, gray, spacing)
                    flagBeats = flags.get(glyph, {})
                    beats = next(iter(flagBeats.values())) if len(flagBeats) == 1 else beatsByGlyph[glyph.name]
                    stem = "up" if len(flagBeats) == 1 and "up" in flagBeats else None
                    if len(flagBeats) == 1 and "down" in flagBeats:
                        stem = "down"
                    if stem is None and glyph.name != "wholeHead":
                        stem = stemDirection(lineInk, glyph, {}, spacing)
                    #the chorales have no dotted notes, so any dot found here is noise
                    placed.append({
                        "glyph": glyph, "staffName": staffName, "beats": beats, "step": step,
                        "confidence": confidence, "score": glyph.score, "stem": stem,
                        "supported": hasLedgerSupport(glyph, system, gray, spacing),
                    })
            voiced, problems = chooseBar(placed, spacing)
            if problems:
                #one unread note must not lose the whole piece: keep what was read, mark it as doubtful,
                #say what is missing, and carry on with the next bar at its proper time
                warnings.warn(f"{barName}: " + "; ".join(problems))
                voiced = [(onset, event._replace(confidence=round(event.confidence * 0.5, 3)), voiceName)
                          for onset, event, voiceName in voiced]
            altered = {}
            for onset, event, voiceName in sorted(voiced, key=lambda item: item[1].x):
                glyph = next(item["glyph"] for item in placed if item["glyph"].cx == event.x and item["glyph"].cy == event.y)
                step = event.pitch
                #the renderer keeps accidentals per voice, until the end of the bar
                if glyph in accidentals:
                    altered[(voiceName, step)] = accidentals[glyph]
                alter = altered.get((voiceName, step), keyAlters(keyCount, keyAlter).get(step % 7, 0))
                start = round((barStart + onset) * ticksPerBeat)
                notes.append(Note(stepToPitch(step) + alter, start, start + round(event.beats * ticksPerBeat), event.confidence))
            barNumber += 1
    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes, ticksPerBeat
