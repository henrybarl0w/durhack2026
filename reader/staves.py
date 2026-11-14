from collections import namedtuple

import cv2
import numpy

from .errors import ReaderError

#a staff: five line centres (top to bottom), its line thickness and its left and right ends
Staff = namedtuple("Staff", ["lines", "thickness", "left", "right"])
System = namedtuple("System", ["treble", "bass", "barlines"])

lightLevel = 200
inkLevel = 160
#anything visibly darker than the white page, so even faint antialiased lines count
faintLevel = 245


def loadGray(imagePath):
    image = cv2.imread(str(imagePath), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ReaderError("could not read image: " + str(imagePath))
    if image.dtype != numpy.uint8:
        image = cv2.convertScaleAbs(image, alpha=255.0 / numpy.iinfo(image.dtype).max)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4].astype(numpy.float32) / 255
        image = (image[:, :, :3] * alpha + 255 * (1 - alpha)).astype(numpy.uint8)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def longestRuns(mask):
    #for every row: length, start and end of its longest horizontal run of true pixels
    height, width = mask.shape
    padded = numpy.zeros((height, width + 2), numpy.int8)
    padded[:, 1:-1] = mask
    changes = numpy.diff(padded, axis=1)
    lengths = numpy.zeros(height, int)
    starts = numpy.zeros(height, int)
    for row in range(height):
        runStarts = numpy.flatnonzero(changes[row] == 1)
        if len(runStarts) == 0:
            continue
        runLengths = numpy.flatnonzero(changes[row] == -1) - runStarts
        best = int(numpy.argmax(runLengths))
        lengths[row], starts[row] = runLengths[best], runStarts[best]
    return lengths, starts, starts + lengths


def findLines(gray, runs, minimumLength):
    #staff lines are the rows holding a long horizontal run of ink
    lengths, starts, ends = runs
    isLine = lengths >= minimumLength
    darkness = 255.0 - gray
    lines = []
    row = 0
    while row < len(isLine):
        if not isLine[row]:
            row += 1
            continue
        first = row
        while row < len(isLine) and isLine[row]:
            row += 1
        rows = numpy.arange(first, row)
        left, right = int(starts[rows].min()), int(ends[rows].max())
        weights = darkness[rows, left:right].sum(axis=1)
        centre = float((rows * weights).sum() / weights.sum())
        thickness = float(weights.sum() / (255.0 * (right - left)))
        lines.append((centre, thickness, left, right))
    return lines


def groupStaves(lines, spacing=None):
    staves = []
    index = 0
    while index + 5 <= len(lines):
        group = lines[index:index + 5]
        gaps = numpy.diff([line[0] for line in group])
        isEven = gaps.min() > 0 and gaps.max() < 1.25 * gaps.min()
        if isEven and (spacing is None or abs(numpy.median(gaps) / spacing - 1) < 0.15):
            staves.append(Staff(
                [line[0] for line in group],
                float(numpy.mean([line[1] for line in group])),
                min(line[2] for line in group),
                max(line[3] for line in group),
            ))
            index += 5
        else:
            index += 1
    if not staves:
        raise ReaderError("no five-line staves found in the image")
    return staves


def staffSpacing(staves):
    return float(numpy.median([gap for staff in staves for gap in numpy.diff(staff.lines)]))


def roughSpacing(gray):
    #a first guess at the staff spacing from the longest lines on the page, only used to size the clef search
    runs = longestRuns(gray < faintLevel)
    if runs[0].max() == 0:
        raise ReaderError("the image is blank")
    return runs, staffSpacing(groupStaves(findLines(gray, runs, 0.5 * runs[0].max())))


def findStavesByRuns(gray, runs, spacing):
    #the old page-wide way: any row holding a long run of ink is a staff line candidate.
    #used only when no clef pairs are found
    staves = groupStaves(findLines(gray, runs, 6 * spacing), spacing)
    if len(staves) % 2:
        raise ReaderError(f"found {len(staves)} staves, a grand staff score needs them in pairs")
    return staves


def clefPairs(gray, spacing):
    #every line of music starts with a treble clef above a bass clef: find them, and pair each treble clef
    #with the bass clef straight below it
    from .glyphs import matchAll, scaledTemplates

    ink = (255 - gray).astype(numpy.float32)
    templates = scaledTemplates(spacing)
    trebles = matchAll(ink, "gClef", templates["gClef"])
    basses = matchAll(ink, "fClef", templates["fClef"])
    pairs = []
    for treble in sorted(trebles, key=lambda clef: -clef.score):
        below = [
            bass for bass in basses
            if abs(bass.cx - treble.cx) < 4 * spacing and 3 * spacing < bass.cy - treble.cy < 20 * spacing
        ]
        if not below:
            continue
        bass = min(below, key=lambda clef: clef.cy - treble.cy)
        #a weaker duplicate of a clef already used is not a new line of music
        if any(abs(treble.cy - other.cy) < 4 * spacing and abs(treble.cx - other.cx) < 4 * spacing for other, _ in pairs):
            continue
        pairs.append((treble, bass))
    return sorted(pairs, key=lambda pair: pair[0].cy)


def rowCoverage(dark, top, bottom, left, right):
    #for each row from top to bottom: how much of the columns from left to right are ink, 0 to 1
    top, bottom = max(0, int(top)), min(dark.shape[0], int(bottom))
    left, right = max(0, int(left)), min(dark.shape[1], int(right))
    if top >= bottom or left >= right:
        return numpy.arange(top, top), numpy.zeros(0)
    return numpy.arange(top, bottom), dark[top:bottom, left:right].mean(axis=1)


def linesBesideClef(dark, clef, spacing):
    #in a strip just right of the clef, staff lines are rows of ink from end to end; the key and time
    #signature there only cover part of a row. take the five evenly spaced rows that best fit the clef
    left, right = clef.cx + clef.width / 2, clef.cx + clef.width / 2 + 8 * spacing
    rows, coverage = rowCoverage(dark, clef.cy - clef.height, clef.cy + clef.height, left, right)
    candidates = []
    row = 0
    while row < len(rows):
        if coverage[row] < 0.7:
            row += 1
            continue
        first = row
        while row < len(rows) and coverage[row] >= 0.7:
            row += 1
        band = numpy.arange(first, row)
        weights = coverage[band]
        candidates.append(float((rows[band] * weights).sum() / weights.sum()))
    best = None
    for start in range(len(candidates) - 4):
        group = candidates[start:start + 5]
        gaps = numpy.diff(group)
        if gaps.min() <= 0 or gaps.max() > 1.25 * gaps.min() or abs(numpy.median(gaps) / spacing - 1) > 0.25:
            continue
        #the clef sits on its staff: prefer the five lines whose middle is nearest the clef's middle
        miss = abs((group[0] + group[4]) / 2 - clef.cy)
        if best is None or miss < best[0]:
            best = (miss, group)
    return None if best is None else best[1]


def staffAt(gray, runs, dark, clef, spacing):
    lines = linesBesideClef(dark, clef, spacing)
    if lines is None:
        return None
    #ends and thickness the way the page-wide method measured them (longest run of ink along a row, and how
    #dark the line's rows are), but only on the rows of these five lines, so nothing elsewhere can interfere
    lengths, starts, ends = runs
    darkness = 255.0 - gray
    band = max(1, int(round(0.3 * spacing)))
    lefts, rights, thicknesses = [], [], []
    for y in lines:
        rows = numpy.arange(max(0, int(round(y)) - band), min(gray.shape[0], int(round(y)) + band + 1))
        best = rows[numpy.argmax(lengths[rows])]
        left, right = int(starts[best]), int(ends[best])
        lefts.append(left)
        rights.append(right)
        lineRows = rows[lengths[rows] >= 0.5 * lengths[best]]
        thicknesses.append(float(darkness[lineRows, left:right].sum() / (255.0 * max(1, right - left))))
    #the median of the five, so one line spoiled by a smudge or glare cannot move the ends
    return Staff(list(lines), float(numpy.median(thicknesses)), int(numpy.median(lefts)), int(numpy.median(rights)))


def findStaves(gray):
    #anchored on the clefs: the staff lines are measured right beside each clef, where they are always
    #clean, then followed to the right. long horizontal shapes elsewhere on the page (glare edges, page
    #edges) never get a say, and a line of music only one bar long is found like any other.
    runs, spacing = roughSpacing(gray)
    dark = gray < faintLevel
    staves = []
    for treble, bass in clefPairs(gray, spacing):
        trebleStaff, bassStaff = staffAt(gray, runs, dark, treble, spacing), staffAt(gray, runs, dark, bass, spacing)
        if trebleStaff is not None and bassStaff is not None:
            staves += [trebleStaff, bassStaff]
    #a line of music whose clef was not found (glare, blur) is looked for the old way, and kept if it does not
    #overlap a line already found from its clefs
    try:
        byRuns = findStavesByRuns(gray, runs, spacing)
    except ReaderError:
        byRuns = []
    for treble, bass in zip(byRuns[0::2], byRuns[1::2]):
        top, bottom = treble.lines[0], bass.lines[4]
        if not any(other.lines[0] <= bottom + spacing and other.lines[4] >= top - spacing for other in staves):
            staves += [treble, bass]
    if not staves:
        raise ReaderError("no five-line staves found in the image")
    staves.sort(key=lambda staff: staff.lines[0])
    return staves, staffSpacing(staves)


def findBarlines(ink, treble, bass, spacing):
    #barlines run unbroken from the top line of the treble staff to the bottom line of the bass staff
    top, bottom = int(round(treble.lines[0])), int(round(bass.lines[4]))
    left, right = min(treble.left, bass.left), max(treble.right, bass.right)
    #a photographed barline can lean or break a little: let it wander a fifth of a space sideways
    reach = max(1, int(round(0.2 * spacing))) * 2 + 1
    band = cv2.dilate(ink[top:bottom + 1, left:right + 1].astype(numpy.uint8), numpy.ones((1, reach), numpy.uint8))
    coverage = band.mean(axis=0)
    columns = numpy.flatnonzero(coverage > 0.9) + left
    barlines = []
    for column in columns:
        if barlines and column - barlines[-1][1] <= max(2, 0.8 * spacing):
            barlines[-1][1] = column
        else:
            barlines.append([column, column])
    if not barlines:
        raise ReaderError("could not find the barlines of a system")
    return [(start + end) / 2 for start, end in barlines]


def eraseLines(gray, staves, spacing, ledgerCount=7):
    #whiten staff and ledger lines wherever nothing crosses them, so only the symbols are left
    cleaned = gray.copy()
    height = gray.shape[0]
    for staff in staves:
        half = staff.thickness / 2 + 1
        positions = list(staff.lines)
        positions += [staff.lines[0] - spacing * step for step in range(1, ledgerCount)]
        positions += [staff.lines[4] + spacing * step for step in range(1, ledgerCount)]
        left, right = staff.left, staff.right + 1
        for centre in positions:
            top, bottom = int(numpy.floor(centre - half)), int(numpy.ceil(centre + half))
            if top - 1 < 0 or bottom + 1 >= height:
                continue
            above = gray[top - 1, left:right] > lightLevel
            below = gray[bottom + 1, left:right] > lightLevel
            clear = above & below
            cleaned[top:bottom + 1, left:right][:, clear] = 255
    return cleaned
