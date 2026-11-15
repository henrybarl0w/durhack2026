import json
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple
from pathlib import Path

import cv2
import numpy

from . import features

templatesFolder = Path(__file__).resolve().parent / "templates"

#cx and cy are the centre of the matched box, everything in pixels of the image being read
Match = namedtuple("Match", ["name", "score", "cx", "cy", "width", "height"])

#lowest normalised correlation accepted for each glyph
thresholds = {
    "wholeHead": 0.6,
    "halfHead": 0.6,
    "blackHead": 0.6,
    "blockRest": 0.72,
    "quarterRest": 0.65,
    "eighthRest": 0.65,
    "sixteenthRest": 0.65,
    "flag8Up": 0.5,
    "flag8Down": 0.5,
    "flag16Up": 0.5,
    "flag16Down": 0.5,
    "sharp": 0.72,
    "natural": 0.72,
    #0.72 threw away the key signature's flats on a blurred two-flat page (they scored 0.68 to 0.71), reading the key as none and every B and E a semitone off
    "flat": 0.66,
    "gClef": 0.55,
    "fClef": 0.55,
    "four": 0.55,
    "dot": 0.9,
}
#a sharp scoring between this and its threshold counts when a notehead sits just to its right (the weakSharps step)
weakSharpScore = 0.58
#match the templates side by side (see findGlyphs)
parallelMatching = True
matchThreads = 4
#a sharp scoring between this and weakSharpScore is accepted only where a notehead sits right after it at almost the same height (the spot an
#accidental belongs): on a blurred page most sharps score here, and few other things in this spot look like one
#(0.48 was tried on three saved scans: it found 2 more sharps in 26 and added 3 to 7 false ones, and chords exact fell 54 to 50 and 51 to 45, so
#it is set equal to weakSharpScore, which leaves this lower tier switched off)
lowestSharpScore = 0.58
headNames = ("wholeHead", "halfHead", "blackHead")
restNames = ("blockRest", "quarterRest", "eighthRest", "sixteenthRest")
flagNames = ("flag8Up", "flag8Down", "flag16Up", "flag16Down")
accidentalNames = ("sharp", "natural", "flat")
furnitureNames = ("gClef", "fClef", "four")

_cache = {}


_lock = threading.Lock()


def loadTemplates():
    #read by several threads at once when the page is read more than once together, so the first load takes a lock and the cache is only
    #ever seen complete
    with _lock:
        if not _cache:
            info = json.loads((templatesFolder / "templates.json").read_text(encoding="utf-8"))
            images = {name: cv2.imread(str(templatesFolder / (name + ".png")), cv2.IMREAD_GRAYSCALE) for name in info["glyphs"]}
            _cache["images"] = images
            _cache["spacing"] = info["staffSpacing"]
    return _cache["spacing"], _cache["images"]


def scaledTemplates(spacing):
    templateSpacing, images = loadTemplates()
    factor = spacing / templateSpacing
    scaled = {}
    for name, image in images.items():
        size = (max(3, round(image.shape[1] * factor)), max(3, round(image.shape[0] * factor)))
        scaled[name] = cv2.resize(image, size, interpolation=cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC)
    return scaled


def peaks(result, threshold, width, height):
    #local maxima of a match map, at least about half a glyph apart
    kernel = numpy.ones((max(1, height // 2) | 1, max(1, width // 2) | 1), numpy.uint8)
    isPeak = (result >= threshold) & (result == cv2.dilate(result, kernel))
    rows, columns = numpy.nonzero(isPeak)
    return [(float(result[row, column]), column, row) for row, column in zip(rows, columns)]


def overlap(first, second):
    dx = min(first.cx + first.width / 2, second.cx + second.width / 2) - max(first.cx - first.width / 2, second.cx - second.width / 2)
    dy = min(first.cy + first.height / 2, second.cy + second.height / 2) - max(first.cy - first.height / 2, second.cy - second.height / 2)
    if dx <= 0 or dy <= 0:
        return 0.0
    return dx * dy / min(first.width * first.height, second.width * second.height)


def matchAll(ink, name, template, floor=1.0):
    #floor below 1 also returns weaker matches than the glyph's threshold (only the debug trace asks for those)
    height, width = template.shape
    if height > ink.shape[0] or width > ink.shape[1]:
        return []
    result = cv2.matchTemplate(ink, (255 - template).astype(numpy.float32), cv2.TM_CCOEFF_NORMED)
    result = numpy.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return [
        Match(name, score, column + width / 2, row + height / 2, width, height)
        for score, column, row in peaks(result, thresholds[name] * floor, width, height)
    ]


def findGlyphs(cleaned, spacing, trace=None):
    #matches every template on the line-free image and keeps the best glyph wherever two overlap;
    #augmentation dots come back separately, as only a dot right after a note or rest means anything.
    #trace, if given, is filled with what was NOT kept and why, for the debug dump: matches that fell a little
    #short of their glyph's threshold ("nearMisses") and matches beaten by a better one on the same spot ("suppressed")
    ink = (255 - cleaned).astype(numpy.float32)
    templates = scaledTemplates(spacing)
    floor = 1.0 if trace is None else 0.8
    weakSharps = features.on("weakSharps")
    candidates = []
    jobs = []
    for name, template in templates.items():
        if name == "dot":
            continue
        reach = min(floor, lowestSharpScore / thresholds[name]) if (weakSharps and name == "sharp") else floor
        jobs.append((name, template, reach))
    if parallelMatching:
        #OpenCV's matching lets go of the interpreter's lock, so the templates can be matched side by side; results are collected in the
        #templates' own order, so the candidates (and everything after) come out exactly as they do one at a time
        with ThreadPoolExecutor(max_workers=matchThreads) as pool:
            for found in pool.map(lambda job: matchAll(ink, *job), jobs):
                candidates.extend(found)
    else:
        for job in jobs:
            candidates.extend(matchAll(ink, *job))
    candidates.sort(key=lambda match: -match.score)
    kept = []
    held = []
    for candidate in candidates:
        if weakSharps and candidate.name == "sharp" and lowestSharpScore <= candidate.score < thresholds["sharp"]:
            held.append(candidate)
            continue
        if candidate.score < thresholds[candidate.name]:
            if trace is not None and candidate.name not in furnitureNames:
                trace.setdefault("nearMisses", []).append(
                    {"name": candidate.name, "score": round(float(candidate.score), 3), "need": thresholds[candidate.name],
                     "x": round(float(candidate.cx), 1), "y": round(float(candidate.cy), 1)})
            continue
        winner = next((other for other in kept if overlap(candidate, other) >= 0.4), None)
        if winner is None:
            kept.append(candidate)
        elif trace is not None:
            trace.setdefault("suppressed", []).append(
                {"name": candidate.name, "score": round(float(candidate.score), 3), "x": round(float(candidate.cx), 1),
                 "y": round(float(candidate.cy), 1), "by": winner.name, "byScore": round(float(winner.score), 3)})
    #step 11 (weakSharps): a blurred sharp scores under its threshold, but a sharp is always written just before its notehead, so a
    #weaker match is accepted when a head sits just to its right on about the same line or space
    for candidate in held:
        if candidate.score >= weakSharpScore:
            near = any(head.name in headNames and 0 < head.cx - candidate.cx < 2.2 * spacing and abs(head.cy - candidate.cy) <= 1.6 * spacing
                       for head in kept)
        else:
            near = any(head.name in headNames and 0.4 * spacing < head.cx - candidate.cx < 1.9 * spacing and abs(head.cy - candidate.cy) <= 0.4 * spacing
                       for head in kept)
        if near and all(overlap(candidate, other) < 0.4 for other in kept):
            kept.append(candidate)
        elif trace is not None:
            trace.setdefault("nearMisses", []).append(
                {"name": "sharp", "score": round(float(candidate.score), 3), "need": thresholds["sharp"],
                 "x": round(float(candidate.cx), 1), "y": round(float(candidate.cy), 1)})
    dots = [dot for dot in matchAll(ink, "dot", templates["dot"]) if all(overlap(dot, other) < 0.2 for other in kept)]
    return kept, dots
