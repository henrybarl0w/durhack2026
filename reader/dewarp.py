#bends a curved page flat again, so every staff line is level from end to end.
#a page held up, lying on a curve or curling at a corner bows: a staff line can drift by half a space or more
#across the page, and the top lines of music can slope while the bottom ones do not. the rest of the reader
#takes each staff line as one height, so that would cut staves short (losing the barlines past the cut),
#move notes by a step, and on a strongly curled page lose whole lines of music.
#the staves are found here without the clefs: the page is cut into narrow upright strips, each strip is
#searched for five evenly spaced lines (which works on a curve or a slope, since a strip is short), and the
#finds are linked from strip to strip into one track per staff. each track is then bent level with one remap.
import cv2
import numpy


def lineKernel(length, degrees):
    #a one-pixel line of the given length and slope, as a morphology kernel
    length = max(5, int(length))
    rise = int(round(length * numpy.tan(numpy.radians(abs(degrees)))))
    kernel = numpy.zeros((rise + 1, length), numpy.uint8)
    for x in range(length):
        y = int(round(x * rise / max(1, length - 1)))
        kernel[rise - y if degrees > 0 else y, x] = 1
    return kernel


def lineImage(gray, length, slopes=(-8, -6, -4, -2, 0, 2, 4, 6, 8)):
    #only long thin strokes that run level or gently sloped: staff lines (and the odd flat beam). stems,
    #noteheads and text are shorter. a curled page slopes its top lines by several degrees, which a level-only
    #search would wipe out, so the strokes are looked for at a few slopes and the finds combined
    #a soft photo greys the lines, so "ink" is measured against the page's own paper level, not a fixed grey
    ink = (gray < numpy.median(gray) - 20).astype(numpy.uint8)
    found = numpy.zeros_like(ink)
    for degrees in slopes:
        found = numpy.maximum(found, cv2.morphologyEx(ink, cv2.MORPH_OPEN, lineKernel(length, degrees)))
    return found.astype(numpy.float32)


def lineSpacing(lines):
    #going down a column of the line picture, the commonest distance from one line to the next is the spacing
    distances = []
    for column in numpy.linspace(0, lines.shape[1] - 1, min(300, lines.shape[1])).astype(int):
        rows = numpy.flatnonzero(lines[:, column] > 0)
        if len(rows) < 2:
            continue
        starts = rows[numpy.concatenate(([True], numpy.diff(rows) > 1))]
        distances += list(numpy.diff(starts))
    distances = [d for d in distances if d >= 4]
    if not distances:
        return None
    return float(numpy.bincount(distances).argmax())


def combProfile(profile, spacing):
    #for every row: how much line ink sits on that row and the four evenly spaced rows below it
    #a sloped line spreads over a few rows of a strip
    padded = numpy.maximum.reduce([numpy.roll(profile, shift) for shift in (-2, -1, 0, 1, 2)])
    score = numpy.zeros_like(profile)
    for k in range(5):
        offset = int(round(k * spacing))
        score[:len(profile) - offset] += padded[offset:]
    return score


def stripFinds(lines, spacing, strip):
    #(middle column, [top line heights of the staves found in that strip]) for each strip
    height, width = lines.shape
    finds = []
    for left in range(0, width - strip + 1, strip):
        profile = lines[:, left:left + strip].mean(axis=1)
        best = numpy.zeros(height, numpy.float32)
        bestSpacing = numpy.full(height, spacing, numpy.float32)
        #perspective can make the spacing differ a little across the page
        for factor in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2):
            score = combProfile(profile, spacing * factor)
            better = score > best
            best[better], bestSpacing[better] = score[better], spacing * factor
        tops = []
        #most of the five lines must be inked across the strip. the comb moved by a line or two still matches
        #most of the same staff, so the best match wins and nothing else within a staff's height is a staff
        for row in numpy.argsort(best)[::-1]:
            if best[row] < 3.0:
                break
            if all(abs(row - other) > 4.5 * spacing for other in tops):
                tops.append(int(row))
        finds.append((left + strip // 2, sorted(tops)))
    return finds


def linkTracks(finds, spacing, strip, width):
    #join finds from neighbouring strips into tracks, one per staff; a strip or two may miss a staff
    tracks = []
    for column, tops in finds:
        claimed = set()
        for track in tracks:
            lastColumn, lastTop = track[-1]
            if column - lastColumn > 3 * strip:
                continue
            near = [top for top in tops if top not in claimed and abs(top - lastTop) <= 0.6 * spacing]
            if near:
                top = min(near, key=lambda value: abs(value - lastTop))
                track.append((column, top))
                claimed.add(top)
        for top in tops:
            if top not in claimed:
                tracks.append([(column, top)])
    #a real staff runs across a good part of the page
    return [track for track in tracks if track[-1][0] - track[0][0] >= 0.15 * width and len(track) >= 4]


def straightenStaves(gray):
    #returns the picture with every staff bent level, or the picture unchanged if no staves can be followed
    height, width = gray.shape
    lines = lineImage(gray, width / 40)
    spacing = lineSpacing(lines)
    if spacing is None:
        return gray
    #the line picture again with a kernel sized from the spacing, so short systems and broken lines stay
    lines = lineImage(gray, 3 * spacing)
    strip = max(8, int(round(2 * spacing)))
    tracks = linkTracks(stripFinds(lines, spacing, strip), spacing, strip, width)
    if not tracks:
        return gray
    columns = numpy.arange(width, dtype=numpy.float32)
    anchors = []  #(where the staff is put, how far its top line sits from there in each column)
    for track in tracks:
        xs = numpy.array([column for column, _ in track], numpy.float32)
        ys = numpy.array([top for _, top in track], numpy.float32)
        #a median over neighbouring strips throws out a strip pulled off course by a beam or a row of notes
        ys = numpy.array([numpy.median(ys[max(0, i - 2):i + 3]) for i in range(len(ys))], numpy.float32)
        level = float(numpy.median(ys))
        drift = numpy.interp(columns, xs, ys - level).astype(numpy.float32)
        anchors.append((level + 2 * spacing, drift))
    anchors.sort(key=lambda anchor: anchor[0])
    #two finds of one staff (a track that broke in two) would fight: keep the longer-reaching one
    merged = []
    for middle, drift in anchors:
        if merged and middle - merged[-1][0] < 2 * spacing:
            continue
        merged.append((middle, drift))
    middles = numpy.array([middle for middle, _ in merged], numpy.float32)
    drifts = numpy.stack([drift for _, drift in merged])
    rows = numpy.arange(height, dtype=numpy.float32)
    #between two staves the drift is blended from both; above the first and below the last it is held
    if len(middles) == 1:
        field = numpy.broadcast_to(drifts[0], (height, width))
    else:
        upper = numpy.clip(numpy.searchsorted(middles, rows) - 1, 0, len(middles) - 2)
        share = numpy.clip((rows - middles[upper]) / (middles[upper + 1] - middles[upper]), 0, 1)[:, None]
        field = drifts[upper] * (1 - share) + drifts[upper + 1] * share
    mapX = numpy.broadcast_to(columns, (height, width)).astype(numpy.float32)
    mapY = (rows[:, None] + field).astype(numpy.float32)
    return cv2.remap(gray, mapX, mapY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
