import json
from collections import namedtuple
from pathlib import Path

import cv2
import numpy

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
    "flat": 0.72,
    "gClef": 0.55,
    "fClef": 0.55,
    "four": 0.55,
    "dot": 0.9,
}
headNames = ("wholeHead", "halfHead", "blackHead")
restNames = ("blockRest", "quarterRest", "eighthRest", "sixteenthRest")
flagNames = ("flag8Up", "flag8Down", "flag16Up", "flag16Down")
accidentalNames = ("sharp", "natural", "flat")
furnitureNames = ("gClef", "fClef", "four")

_cache = {}


def loadTemplates():
    if not _cache:
        info = json.loads((templatesFolder / "templates.json").read_text(encoding="utf-8"))
        _cache["spacing"] = info["staffSpacing"]
        _cache["images"] = {
            name: cv2.imread(str(templatesFolder / (name + ".png")), cv2.IMREAD_GRAYSCALE) for name in info["glyphs"]
        }
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
    candidates = [match for name, template in templates.items() if name != "dot" for match in matchAll(ink, name, template, floor)]
    candidates.sort(key=lambda match: -match.score)
    kept = []
    for candidate in candidates:
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
    dots = [dot for dot in matchAll(ink, "dot", templates["dot"]) if all(overlap(dot, other) < 0.2 for other in kept)]
    return kept, dots
