noteNames = ["C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B"]
sharpSemitones = {1, 3, 6, 8, 10}
unitsPerBeat = 4
unitsPerMeasure = 16
trebleStartPitch = 60
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

attributes = f"""<attributes>
<divisions>{unitsPerBeat}</divisions>
<key><fifths>0</fifths></key>
<time><beats>4</beats><beat-type>4</beat-type></time>
<staves>2</staves>
<clef number="1"><sign>G</sign><line>2</line></clef>
<clef number="2"><sign>F</sign><line>4</line></clef>
</attributes>
"""


def decompose(units):
    parts = []
    for size in sizes:
        while units >= size:
            parts.append(size)
            units -= size
    return parts


def buildEvents(notes, ticksPerBeat):
    unitTicks = ticksPerBeat / unitsPerBeat
    byStart = {}
    for note in notes:
        start = round(note.start / unitTicks)
        end = round(note.end / unitTicks)
        if start not in byStart or note.pitch > byStart[start][2]:
            byStart[start] = (start, end, note.pitch)
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


def layOut(events):
    measures = [[]]
    space = unitsPerMeasure
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
                space = unitsPerMeasure
        for number, (measureIndex, part) in enumerate(parts):
            isPitched = pitch is not None
            measures[measureIndex].append((pitch, part, isPitched and number > 0, isPitched and number < len(parts) - 1))
    if space == unitsPerMeasure and len(measures) > 1:
        measures.pop()
    elif space < unitsPerMeasure:
        for part in decompose(space):
            measures[-1].append((None, part, False, False))
    return measures


def noteXml(pitch, units, staff, tiedFromPrevious, tiedToNext):
    typeName, isDotted = sizeTypes[units]
    isSharp = pitch % 12 in sharpSemitones
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
        f"<note><pitch><step>{noteNames[pitch % 12]}</step>"
        f"{'<alter>1</alter>' if isSharp else ''}<octave>{pitch // 12 - 1}</octave></pitch>"
        f"<duration>{units}</duration>{ties}<voice>{staff}</voice><type>{typeName}</type>"
        f"{'<dot/>' if isDotted else ''}"
        f"{'<accidental>sharp</accidental>' if isSharp and not tiedFromPrevious else ''}"
        f"<staff>{staff}</staff>{notations}</note>\n"
    )


def restXml(units, staff):
    typeName, isDotted = sizeTypes[units]
    return (
        f"<note><rest/><duration>{units}</duration><voice>{staff}</voice><type>{typeName}</type>"
        f"{'<dot/>' if isDotted else ''}<staff>{staff}</staff></note>\n"
    )


def buildMusicXml(notes, ticksPerBeat):
    measures = layOut(buildEvents(notes, ticksPerBeat))
    xml = header
    for number, measure in enumerate(measures, start=1):
        xml += f'<measure number="{number}">\n'
        if number == 1:
            xml += attributes
        for staff in (1, 2):
            if staff == 2:
                xml += f"<backup><duration>{unitsPerMeasure}</duration></backup>\n"
            for pitch, units, tiedFromPrevious, tiedToNext in measure:
                if pitch is not None and (pitch >= trebleStartPitch) == (staff == 1):
                    xml += noteXml(pitch, units, staff, tiedFromPrevious, tiedToNext)
                else:
                    xml += restXml(units, staff)
        xml += "</measure>\n"
    return xml + "</part>\n</score-partwise>\n"
