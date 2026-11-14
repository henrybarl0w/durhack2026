from itertools import combinations

from .errors import RendererError

unitsPerBeat = 4
unitsPerWholeNote = 16
trebleStartPitch = 60
letters = "CDEFGAB"
whiteSemitones = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
sharpOrder = "FCGDAEB"
flatOrder = "BEADGCF"
majorKeys = {"C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5, "F#": 6, "C#": 7,
             "F": -1, "Bb": -2, "Eb": -3, "Ab": -4, "Db": -5, "Gb": -6, "Cb": -7}
minorKeys = {"Am": 0, "Em": 1, "Bm": 2, "F#m": 3, "C#m": 4, "G#m": 5, "D#m": 6, "A#m": 7,
             "Dm": -1, "Gm": -2, "Cm": -3, "Fm": -4, "Bbm": -5, "Ebm": -6, "Abm": -7}
accidentalNames = {1: "sharp", -1: "flat", 0: "natural"}
sizes = (16, 12, 8, 6, 4, 3, 2, 1)
sizeTypes = {
    16: ("whole", False),
    12: ("half", True),
    8: ("half", False),
    6: ("quarter", True),
    4: ("quarter", False),
    3: ("eighth", True),
    2: ("eighth", False),
    1: ("16th", False),
}

header = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="4.0">
<part-list><score-part id="P1"><part-name print-object="no"></part-name></score-part></part-list>
<part id="P1">
"""


def parseKey(key):
    if key in majorKeys:
        return majorKeys[key], "major"
    if key in minorKeys:
        return minorKeys[key], "minor"
    raise RendererError("unknown key: " + str(key))


def keyAlters(fifths):
    if fifths >= 0:
        return {letter: 1 for letter in sharpOrder[:fifths]}
    return {letter: -1 for letter in flatOrder[:-fifths]}


def measureLength(timeSignature):
    beats, beatType = timeSignature
    if beats <= 0 or beatType <= 0 or (beats * unitsPerWholeNote) % beatType != 0:
        raise RendererError("unsupported time signature: " + str(timeSignature))
    return beats * unitsPerWholeNote // beatType


def attributesXml(fifths, mode, timeSignature):
    return f"""<attributes>
<divisions>{unitsPerBeat}</divisions>
<key><fifths>{fifths}</fifths><mode>{mode}</mode></key>
<time><beats>{timeSignature[0]}</beats><beat-type>{timeSignature[1]}</beat-type></time>
<staves>2</staves>
<clef number="1"><sign>G</sign><line>2</line></clef>
<clef number="2"><sign>F</sign><line>4</line></clef>
</attributes>
"""


def tempoXml(tempo):
    return (
        f'<direction placement="above"><direction-type><words>♩ = {tempo}</words></direction-type>'
        f'<sound tempo="{tempo}"/></direction>\n'
    )


def decompose(units):
    parts = []
    for size in sizes:
        while units >= size:
            parts.append(size)
            units -= size
    return parts


def spell(pitch, fifths, alters):
    semitone = pitch % 12
    letter = None
    for candidate in letters:
        if (whiteSemitones[candidate] + alters.get(candidate, 0)) % 12 == semitone:
            letter, alter = candidate, alters.get(candidate, 0)
            break
    if letter is None:
        whites = [candidate for candidate in letters if whiteSemitones[candidate] == semitone]
        if whites:
            letter, alter = whites[0], 0
        elif fifths >= 0:
            letter, alter = [candidate for candidate in letters if whiteSemitones[candidate] == semitone - 1][0], 1
        else:
            letter, alter = [candidate for candidate in letters if whiteSemitones[candidate] == semitone + 1][0], -1
    octave = pitch // 12 - 1
    if letter == "B" and alter == 1 and semitone == 0:
        octave -= 1
    if letter == "C" and alter == -1 and semitone == 11:
        octave += 1
    return letter, alter, octave


def quantize(notes, ticksPerBeat):
    unitTicks = ticksPerBeat / unitsPerBeat
    items = []
    for note in notes:
        start = round(note.start / unitTicks)
        items.append((start, max(round(note.end / unitTicks), start + 1), note.pitch))
    return items


def mostSounding(items):
    points = sorted([(start, 1) for start, _, _ in items] + [(end, -1) for _, end, _ in items])
    sounding = 0
    most = 1
    for _, change in points:
        sounding += change
        most = max(most, sounding)
    return most


def splitChords(items):
    items = sorted(items, key=lambda item: (item[0], -item[2]))
    voiceCount = mostSounding(items)
    voices = [[] for _ in range(voiceCount)]
    voiceEnds = [0] * voiceCount
    lastPitches = [None] * voiceCount
    index = 0
    while index < len(items):
        start = items[index][0]
        group = []
        while index < len(items) and items[index][0] == start:
            group.append(items[index])
            index += 1
        free = [voice for voice in range(voiceCount) if voiceEnds[voice] <= start]
        group = group[:len(free)]

        def cost(choice):
            return sum(abs(item[2] - lastPitches[voice]) for item, voice in zip(group, choice) if lastPitches[voice] is not None)

        for item, voice in zip(group, min(combinations(free, len(group)), key=cost)):
            voices[voice].append(item)
            voiceEnds[voice] = item[1]
            lastPitches[voice] = item[2]
    return [voice for voice in voices if voice]


def splitVoices(notes, ticksPerBeat):
    notesByVoice = {}
    for note in notes:
        notesByVoice.setdefault(note.voice, []).append(note)
    groups = [splitChords(quantize(groupNotes, ticksPerBeat)) for groupNotes in notesByVoice.values()]

    def averagePitch(groupVoices):
        pitches = [item[2] for voice in groupVoices for item in voice]
        return sum(pitches) / max(1, len(pitches))

    groups.sort(key=lambda groupVoices: -averagePitch(groupVoices))
    return [voice for groupVoices in groups for voice in groupVoices] or [[]]


def buildEvents(items):
    byStart = {}
    for start, end, pitch in items:
        if start not in byStart or pitch > byStart[start][2]:
            byStart[start] = (start, end, pitch)
    ordered = sorted(byStart.values())
    events = []
    cursor = 0
    for index, (start, end, pitch) in enumerate(ordered):
        nextStart = ordered[index + 1][0] if index + 1 < len(ordered) else end
        end = max(start + 1, min(end, nextStart))
        if start > cursor:
            events.append((None, start - cursor))
        events.append((pitch, end - start))
        cursor = end
    return events


def layOut(events, measureUnits):
    measures = [[]]
    space = measureUnits
    for pitch, length in events:
        parts = []
        remaining = length
        while remaining > 0:
            chunk = min(remaining, space)
            for part in decompose(chunk):
                parts.append((len(measures) - 1, part))
            remaining -= chunk
            space -= chunk
            if space == 0:
                measures.append([])
                space = measureUnits
        for number, (measureIndex, part) in enumerate(parts):
            isPitched = pitch is not None
            measures[measureIndex].append((pitch, part, isPitched and number > 0, isPitched and number < len(parts) - 1))
    if space == measureUnits and len(measures) > 1:
        measures.pop()
    elif space < measureUnits:
        for part in decompose(space):
            measures[-1].append((None, part, False, False))
    return measures


def noteXml(spelling, units, staff, voice, stem, accidental, tiedFromPrevious, tiedToNext):
    letter, alter, octave = spelling
    typeName, isDotted = sizeTypes[units]
    ties = ""
    tieds = ""
    if tiedFromPrevious:
        ties += '<tie type="stop"/>'
        tieds += '<tied type="stop"/>'
    if tiedToNext:
        ties += '<tie type="start"/>'
        tieds += '<tied type="start"/>'
    notations = f"<notations>{tieds}</notations>" if tieds else ""
    return (
        f"<note><pitch><step>{letter}</step>"
        f"{f'<alter>{alter}</alter>' if alter else ''}<octave>{octave}</octave></pitch>"
        f"<duration>{units}</duration>{ties}<voice>{voice}</voice><type>{typeName}</type>"
        f"{'<dot/>' if isDotted else ''}"
        f"{f'<accidental>{accidental}</accidental>' if accidental else ''}"
        f"{f'<stem>{stem}</stem>' if stem else ''}"
        f"<staff>{staff}</staff>{notations}</note>\n"
    )


def restXml(units, staff, voice):
    typeName, isDotted = sizeTypes[units]
    return (
        f"<note><rest/><duration>{units}</duration><voice>{voice}</voice><type>{typeName}</type>"
        f"{'<dot/>' if isDotted else ''}<staff>{staff}</staff></note>\n"
    )


def pitchedXml(pitch, units, staff, voice, stem, tiedFromPrevious, tiedToNext, key, state):
    fifths, alters = key
    spelling = spell(pitch, fifths, alters)
    letter, alter, octave = spelling
    current = state.get((letter, octave), alters.get(letter, 0))
    state[(letter, octave)] = alter
    accidental = None if tiedFromPrevious or alter == current else accidentalNames[alter]
    return noteXml(spelling, units, staff, voice, stem, accidental, tiedFromPrevious, tiedToNext)


def voiceXml(measure, staff, voice, stem, key):
    xml = ""
    state = {}
    for pitch, units, tiedFromPrevious, tiedToNext in measure:
        if pitch is None:
            xml += restXml(units, staff, voice)
        else:
            xml += pitchedXml(pitch, units, staff, voice, stem, tiedFromPrevious, tiedToNext, key, state)
    return xml


def singleVoiceMeasureXml(measure, measureUnits, key):
    xml = ""
    for staff in (1, 2):
        if staff == 2:
            xml += f"<backup><duration>{measureUnits}</duration></backup>\n"
        state = {}
        for pitch, units, tiedFromPrevious, tiedToNext in measure:
            if pitch is not None and (pitch >= trebleStartPitch) == (staff == 1):
                xml += pitchedXml(pitch, units, staff, staff, None, tiedFromPrevious, tiedToNext, key, state)
            else:
                xml += restXml(units, staff, staff)
    return xml


def multiVoiceMeasureXml(voiceMeasures, measureUnits, key):
    trebleCount = -(-len(voiceMeasures) // 2)
    xml = ""
    for index, measure in enumerate(voiceMeasures):
        staff = 1 if index < trebleCount else 2
        staffSize = trebleCount if staff == 1 else len(voiceMeasures) - trebleCount
        position = index if staff == 1 else index - trebleCount
        stem = None if staffSize == 1 else ("up" if position % 2 == 0 else "down")
        if index > 0:
            xml += f"<backup><duration>{measureUnits}</duration></backup>\n"
        xml += voiceXml(measure, staff, index + 1, stem, key)
    return xml


def buildMusicXml(notes, ticksPerBeat, timeSignature=(4, 4), key="C", tempo=None):
    measureUnits = measureLength(timeSignature)
    fifths, mode = parseKey(key)
    keyContext = (fifths, keyAlters(fifths))
    voices = splitVoices(notes, ticksPerBeat)
    voiceMeasures = [layOut(buildEvents(voice), measureUnits) for voice in voices]
    measureCount = max(len(measures) for measures in voiceMeasures)
    emptyMeasure = [(None, part, False, False) for part in decompose(measureUnits)]
    for measures in voiceMeasures:
        measures.extend([emptyMeasure] * (measureCount - len(measures)))
    xml = header
    for number in range(measureCount):
        xml += f'<measure number="{number + 1}">\n'
        if number == 0:
            xml += attributesXml(fifths, mode, timeSignature)
            if tempo:
                xml += tempoXml(tempo)
        measuresHere = [measures[number] for measures in voiceMeasures]
        if len(voices) == 1:
            xml += singleVoiceMeasureXml(measuresHere[0], measureUnits, keyContext)
        else:
            xml += multiVoiceMeasureXml(measuresHere, measureUnits, keyContext)
        xml += "</measure>\n"
    return xml + "</part>\n</score-partwise>\n"
