#builds reader/templates from the renderer's own output: run with python -m reader.makeTemplates
#needs the renderer's requirements (verovio, resvg-py, mido), the reader itself only needs opencv and numpy
import json
import re
import sys
import xml.etree.ElementTree as elementTree
from pathlib import Path

import cv2
import numpy

repoFolder = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repoFolder))

import renderer
from renderer.rendering import svgToPng

templatesFolder = Path(__file__).resolve().parent / "templates"
svgNamespace = "http://www.w3.org/2000/svg"
xlinkNamespace = "http://www.w3.org/1999/xlink"
hrefKey = "{" + xlinkNamespace + "}href"
zoom = 4.0
ticks = 480

glyphNames = {
    "E0A2": "wholeHead",
    "E0A3": "halfHead",
    "E0A4": "blackHead",
    "E4E3": "blockRest",
    "E4E5": "quarterRest",
    "E4E6": "eighthRest",
    "E4E7": "sixteenthRest",
    "E240": "flag8Up",
    "E241": "flag8Down",
    "E242": "flag16Up",
    "E243": "flag16Down",
    "E262": "sharp",
    "E261": "natural",
    "E260": "flat",
    "E050": "gClef",
    "E062": "fClef",
    "E084": "four",
}


def calibrationScores():
    #a whole note, halves, quarters and the three accidentals, with rests filling the bass staff
    melody = [(72, 4), (72, 2), (76, 2), (72, 1), (74, 1), (76, 1), (77, 1), (66, 1), (70, 1), (69, 1), (67, 1)]
    notes = []
    tick = 0
    for pitch, beats in melody:
        notes.append((pitch, tick, tick + beats * ticks))
        tick += beats * ticks
    yield renderer.notesToScore(notes, ticks, format="svg")
    yield renderer.notesToScore([(63, 0, ticks)], ticks, format="svg", key="F")
    #a three beat gap is drawn as a dotted half rest
    yield renderer.notesToScore([(72, 0, ticks), (72, 4 * ticks, 5 * ticks)], ticks, format="svg")
    #eighths and sixteenths with stems up and down: the renderer gives each its own flag, never a beam
    short = [(60, 0.5), (62, 0.25), (64, 0.25), (79, 0.5), (77, 0.25), (76, 0.25), (72, 2)]
    notes = []
    tick = 0
    for pitch, beats in short:
        notes.append((pitch, tick, tick + round(beats * ticks)))
        tick += round(beats * ticks)
    yield renderer.notesToScore(notes, ticks, format="svg")


def pixelsPerUnit(root):
    inner = root.find(f"{{{svgNamespace}}}svg")
    viewBoxWidth = float(inner.get("viewBox").split()[2])
    return float(root.get("width").rstrip("px")) / viewBoxWidth


def staffSpacingUnits(svgText):
    lines = re.findall(r'<path d="M\S+ (\d+) L\S+ \1" stroke-width', svgText)
    first = [int(value) for value in lines[:5]]
    return (first[4] - first[0]) / 4


def isolatedGlyph(root, element):
    #a copy of the page with every drawing removed except this one glyph
    copy = elementTree.fromstring(elementTree.tostring(root))
    margin = copy.find(f".//{{{svgNamespace}}}g[@class='page-margin']")
    for child in list(margin):
        margin.remove(child)
    margin.append(elementTree.fromstring(elementTree.tostring(element)))
    png = svgToPng(elementTree.tostring(copy, encoding="unicode"), zoom)
    image = cv2.imdecode(numpy.frombuffer(png, numpy.uint8), cv2.IMREAD_GRAYSCALE)
    ink = numpy.argwhere(image < 250)
    (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
    pad = 2
    return image[max(0, top - pad):bottom + pad + 1, max(0, left - pad):right + pad + 1]


def main():
    elementTree.register_namespace("", svgNamespace)
    elementTree.register_namespace("xlink", xlinkNamespace)
    names = glyphNames
    expected = set(names.values()) | {"dot"}
    found = {}
    spacing = None
    for svgText in calibrationScores():
        root = elementTree.fromstring(svgText)
        spacingPixels = staffSpacingUnits(svgText) * pixelsPerUnit(root) * zoom
        if spacing is not None and abs(spacingPixels - spacing) > 0.01:
            raise RuntimeError("calibration scores disagree on the staff spacing")
        spacing = spacingPixels
        for use in root.iter(f"{{{svgNamespace}}}use"):
            code = use.get(hrefKey, "").lstrip("#").split("-")[0]
            if code in names and names[code] not in found:
                found[names[code]] = isolatedGlyph(root, use)
        #augmentation dots are drawn as plain ellipses rather than font glyphs
        for group in root.iter(f"{{{svgNamespace}}}g"):
            if group.get("class") == "dots" and len(group) and "dot" not in found:
                found["dot"] = isolatedGlyph(root, group)
    missing = sorted(expected - set(found))
    if missing:
        raise RuntimeError("calibration scores did not contain: " + ", ".join(missing))

    templatesFolder.mkdir(exist_ok=True)
    for name, image in found.items():
        cv2.imwrite(str(templatesFolder / (name + ".png")), image)
    info = {"staffSpacing": spacing, "zoom": zoom, "glyphs": sorted(found)}
    (templatesFolder / "templates.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"wrote {len(found)} templates to {templatesFolder} (staff spacing {spacing:.2f}px)")


if __name__ == "__main__":
    main()
