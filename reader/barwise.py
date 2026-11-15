#reads a page one bar at a time, each bar a unit of its own. the page is cut into bars along its barlines; each bar is cropped
#out together with its system's clef and key signature (so it still looks like a line of music and the reader finds its
#staves, clefs and key), straightened, read on its own, and the notes are put back together with each bar timed from its own start.
#the straightening uses nothing but the bar's own pixels. the page is NOT dewarped as a whole: following a line across a
#whole page lets the error pile up, and the page's own bend is no use to a bar. inside one bar the staff lines are close to
#straight, so they are found there (long thin strokes at gentle slopes, then five evenly spaced lines), each staff's top line is
#fitted with a gentle curve across the bar, and the bar is stretched so its ten lines lie level and one standard space apart.
#the clef piece and the bar piece come from different places on the page; each is straightened on its own to the same line
#heights, so they join without a step.
#   notes, ticksPerBeat = readBarwise("score.png")
import os
import tempfile
import warnings

import cv2
import numpy

from . import dewarp
from .errors import ReaderError
from .glyphs import findGlyphs
from .score import Note, beatsPerBar, headerRight, keyConsensus, normalise, pickupLength, readScore, ticksPerBeat
from .staves import System, eraseLines, findBarlines, lightLevel, loadGray


def robustCurve(xs, ys, columns, degree):
    #a gentle curve through the points, found so that a few wrong points (a stem, a note head) cannot pull it
    xs, ys = numpy.asarray(xs, numpy.float64), numpy.asarray(ys, numpy.float64)
    keep = numpy.ones(len(xs), bool)
    coefficients = numpy.polyfit(xs, ys, min(degree, len(xs) - 1))
    for _ in range(3):
        error = numpy.abs(numpy.polyval(coefficients, xs) - ys)
        keep = error <= max(1.0, 2.5 * numpy.median(error[keep]))
        if keep.sum() < 3:
            break
        coefficients = numpy.polyfit(xs[keep], ys[keep], min(degree, keep.sum() - 1))
    return numpy.polyval(coefficients, columns).astype(numpy.float32)


def trackStaves(piece, spacing):
    #the staves in this piece alone: (the spacing found here, every strip find, each staff's top line as a curve across the
    #piece, upper first). fewer than two curves means the two staves cannot be followed in this piece
    height, width = piece.shape
    lines = dewarp.lineImage(piece, 3 * spacing)
    local = dewarp.lineSpacing(lines)
    local = local if local and 0.7 * spacing <= local <= 1.4 * spacing else spacing
    strip = max(8, int(round(2 * local)))
    finds = dewarp.stripFinds(lines, local, strip)
    tracks = dewarp.linkTracks(finds, local, strip, width)
    #the two longest tracks are the two staves
    tracks = sorted(tracks, key=len, reverse=True)[:2]
    columns = numpy.arange(width, dtype=numpy.float32)
    fitted = [robustCurve([c for c, _ in track], [t for _, t in track], columns, 2 if len(track) >= 6 else 1) for track in tracks]
    fitted.sort(key=lambda top: float(top.mean()))
    return local, finds, fitted


def fitLines(piece, top, spacing):
    #the five lines of one staff, each as its own curve across the piece. the top line's curve is only the first guess of where
    #to look: every line is looked for in its own right (the darkest row near where it should be, a short window of columns at
    #a time) and fitted separately, so a spacing that changes across the bar (the page seen at an angle) is followed too
    height, width = piece.shape
    ink = 255.0 - piece.astype(numpy.float32)
    window, step = max(8, int(2 * spacing)), max(4, int(spacing))
    columns = numpy.arange(width, dtype=numpy.float32)
    out = []
    for k in range(5):
        xs, ys = [], []
        for start in range(0, width - window + 1, step):
            centre = start + window // 2
            profile = numpy.median(ink[:, start:start + window], axis=1)
            expected = float(top[centre]) + k * spacing
            low, high = int(max(0, round(expected - 0.4 * spacing))), int(min(height - 1, round(expected + 0.4 * spacing)))
            if high - low < 3:
                continue
            segment = profile[low:high + 1]
            peak = int(segment.argmax())
            if segment[peak] < 1.3 * float(numpy.median(segment)) + 3:
                continue
            y = float(low + peak)
            if 0 < peak < len(segment) - 1:
                before, here, after = float(segment[peak - 1]), float(segment[peak]), float(segment[peak + 1])
                if before - 2 * here + after != 0:
                    y += 0.5 * (before - after) / (before - 2 * here + after)
            xs.append(centre)
            ys.append(y)
        if len(xs) >= 4:
            out.append(robustCurve(xs, ys, columns, 2))
        else:
            out.append((top + k * spacing).astype(numpy.float32))
    return out


def fitStaffTogether(piece, top, spacing):
    #the five lines of one staff fitted as ONE model instead of five separate curves: the top line is a gentle curve across the
    #bar and the gap between lines changes smoothly along it (the page seen at an angle), so every line shares the evidence of all
    #the others. a line hidden by the clef's tail or a run of notes is then carried by the rest, where a separate fit would
    #wander off after the wrong ink. returns the five curves, or None when too little of the staff could be seen
    height, width = piece.shape
    ink = 255.0 - piece.astype(numpy.float32)
    window, step = max(8, int(2 * spacing)), max(4, int(spacing))
    columns = numpy.arange(width, dtype=numpy.float64)
    seen = []
    for k in range(5):
        for start in range(0, width - window + 1, step):
            centre = start + window // 2
            profile = numpy.median(ink[:, start:start + window], axis=1)
            expected = float(top[centre]) + k * spacing
            low, high = int(max(0, round(expected - 0.4 * spacing))), int(min(height - 1, round(expected + 0.4 * spacing)))
            if high - low < 3:
                continue
            segment = profile[low:high + 1]
            peak = int(segment.argmax())
            if segment[peak] < 1.3 * float(numpy.median(segment)) + 3:
                continue
            y = float(low + peak)
            if 0 < peak < len(segment) - 1:
                before, here, after = float(segment[peak - 1]), float(segment[peak]), float(segment[peak + 1])
                if before - 2 * here + after != 0:
                    y += 0.5 * (before - after) / (before - 2 * here + after)
            seen.append((centre, y, k))
    if len(seen) < 10 or len({k for _, _, k in seen}) < 4:
        return None
    xs, ys, ks = (numpy.array(column, numpy.float64) for column in zip(*seen))
    middle = width / 2.0
    u = (xs - middle) / width
    design = numpy.stack([numpy.ones_like(u), u, u * u, ks, ks * u], axis=1)
    keep = numpy.ones(len(xs), bool)
    for _ in range(4):
        coefficients = numpy.linalg.lstsq(design[keep], ys[keep], rcond=None)[0]
        error = numpy.abs(design @ coefficients - ys)
        limit = max(1.0, 2.5 * float(numpy.median(error[keep])))
        newer = error <= limit
        if newer.sum() < 8 or (newer == keep).all():
            break
        keep = newer
    low, high = xs[keep].min(), xs[keep].max()
    inside = numpy.clip(columns, low, high)
    v = (inside - middle) / width
    out = []
    for k in range(5):
        line = coefficients[0] + coefficients[1] * v + coefficients[2] * v * v + k * (coefficients[3] + coefficients[4] * v)
        slope = (coefficients[1] + 2 * coefficients[2] * v + k * coefficients[4]) / width   # carries straight on past the ends
        out.append((line + slope * (columns - inside)).astype(numpy.float32))
    return out


def measureBar(piece, spacing, together=False):
    #everything found in one piece, from its own pixels only: the ten lines as curves, or None if the two staves cannot be followed
    local, finds, tops = trackStaves(piece, spacing)
    if len(tops) < 2:
        return None
    lines = []
    for top in tops:
        staff = fitStaffTogether(piece, top, local) if together else None
        lines += staff if staff is not None else fitLines(piece, top, local)
    return {"spacing": local, "finds": finds, "tops": tops, "lines": lines}


def straightenBar(piece, spacing, canonicalTops=None, together=False):
    #the piece stretched so that each of its ten staff lines, found as a curve of its own, lies level at its place on a flat page:
    #the staves' top lines at canonicalTops ((upper, lower), or where the bar's own lines sit on average) and a standard
    #space between lines. returns the straightened piece and what was found, or the piece unchanged if the staves cannot be followed
    height, width = piece.shape
    found = measureBar(piece, spacing, together)
    if found is None:
        return piece, None
    lines = found["lines"]
    if canonicalTops is None:
        canonicalTops = (float(numpy.median(lines[0])), float(numpy.median(lines[5])))
    source = numpy.stack(lines)                                                            # ten rows, one value per column
    target = numpy.array([canonicalTops[s] + k * spacing for s in range(2) for k in range(5)], numpy.float32)
    rows = numpy.arange(height, dtype=numpy.float32)
    mapY = numpy.empty((height, width), numpy.float32)
    for column in range(width):
        src = numpy.maximum.accumulate(source[:, column])                                  # lines never cross
        mapped = numpy.interp(rows, target, src)
        above, below = rows < target[0], rows > target[-1]
        mapped[above] = src[0] + (rows[above] - target[0])
        mapped[below] = src[-1] + (rows[below] - target[-1])
        mapY[:, column] = mapped
    mapX = numpy.broadcast_to(numpy.arange(width, dtype=numpy.float32), (height, width)).copy()
    flat = cv2.remap(piece, mapX, mapY, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    found["target"] = target
    return flat, found


def layout(imagePath):
    #the page cut into systems and bars: everything the bar by bar reader (and the pictures of it) needs
    gray, staves, spacing = normalise(loadGray(imagePath))
    lineInk = gray < lightLevel
    systems = [System(staves[i], staves[i + 1], findBarlines(lineInk, staves[i], staves[i + 1], spacing))
               for i in range(0, len(staves) - 1, 2)]
    #as the page reader does: a lost closing or opening line is put back at the ends of the staff lines
    for index, system in enumerate(systems):
        start = min(system.treble.left, system.bass.left)
        end = max(system.treble.right, system.bass.right)
        barlines = list(system.barlines)
        if barlines[0] - start > 3 * spacing:
            barlines.insert(0, start)
        if end - barlines[-1] > 3 * spacing:
            barlines.append(end)
        systems[index] = system._replace(barlines=barlines)
    glyphs, _ = findGlyphs(eraseLines(gray, staves, spacing), spacing)
    windows = []
    for index, system in enumerate(systems):
        top = system.treble.lines[0] - 8 * spacing
        bottom = system.bass.lines[4] + 8 * spacing
        if index > 0:
            top = max(top, (systems[index - 1].bass.lines[4] + system.treble.lines[0]) / 2)
        if index + 1 < len(systems):
            bottom = min(bottom, (system.bass.lines[4] + systems[index + 1].treble.lines[0]) / 2)
        windows.append((max(0, int(top)), min(gray.shape[0], int(bottom))))
    inSystems = [[g for g in glyphs if top <= g.cy < bottom] for top, bottom in windows]
    keyCount, _ = keyConsensus(inSystems, systems, spacing, None)
    headerEnds = [headerRight(inSystems[i], systems[i], spacing, keyCount) for i in range(len(systems))]
    return {"gray": gray, "spacing": spacing, "systems": systems, "windows": windows, "headerEnds": headerEnds}


def barPieces(page, index, flatten=True, together=False):
    #the system's bars as pictures, each ready to read: [(bar left, bar right, picture, what the straightening found)]
    gray, spacing, system, (top, bottom) = page["gray"], page["spacing"], page["systems"][index], page["windows"][index]
    band = gray[top:bottom]
    headerEnd = page["headerEnds"][index]
    canonical = (system.treble.lines[0] - top, system.bass.lines[0] - top)

    def cut(x0, x1):
        x0, x1 = max(0, int(x0)), int(min(band.shape[1], x1))
        piece = band[:, x0:x1]
        if not flatten:
            return piece, None
        return straightenBar(piece, spacing, canonical, together)

    out = []
    bars = list(zip(system.barlines, system.barlines[1:]))
    for j, (left, right) in enumerate(bars):
        closing = right + 0.7 * spacing
        if j == 0:
            picture, found = cut(system.barlines[0] - spacing, closing)
        else:
            head, _ = cut(system.barlines[0] - spacing, headerEnd + 0.5 * spacing)
            body, found = cut(left - 0.35 * spacing, closing)
            picture = numpy.hstack([head, body])
        out.append((left, right, picture, found))
    return out


def readBarwise(imagePath, keepFolder=None, flatten=True, together=False):
    page = layout(imagePath)
    spacing = page["spacing"]
    work = keepFolder or tempfile.mkdtemp(prefix="barwise-")
    os.makedirs(work, exist_ok=True)
    notes = []
    clock = 0.0
    barIndex = 0
    for index, system in enumerate(page["systems"]):
        headerEnd = page["headerEnds"][index]
        bars = list(zip(system.barlines, system.barlines[1:]))
        widths = sorted(right - left for left, right in bars)
        typical = widths[len(widths) // 2] if widths else 0
        for left, right, picture, _ in barPieces(page, index, flatten, together):
            crop = cv2.copyMakeBorder(picture, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
            path = os.path.join(work, f"bar{barIndex + 1:02d}.png")
            cv2.imwrite(path, crop)
            barLength = float(beatsPerBar)
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    read, ticks = readScore(path)
                #a first bar of the piece that is narrow and short in every voice is a pickup, not a bar with notes missing
                if barIndex == 0:
                    problems = [str(w.message) for w in caught]
                    narrow = (right - max(left, headerEnd)) < 0.6 * typical if len(bars) > 1 else False
                    if narrow and pickupLength(problems):
                        barLength = pickupLength(problems)
                for note in read:
                    start = round(clock * ticksPerBeat) + note.start
                    notes.append(Note(note.pitch, start, start + (note.end - note.start), note.confidence))
            except ReaderError:
                pass
            clock += barLength
            barIndex += 1
    if not notes:
        raise ReaderError("no notes could be read from any bar")
    notes.sort(key=lambda note: (note.start, note.pitch))
    return notes, ticksPerBeat
