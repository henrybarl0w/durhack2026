#draws what the reader sees on an image: staff lines, barlines, and every symbol it matched with its score
#run from the repo folder:
#   python reader/tests/showSeen.py score.png
#   python reader/tests/showSeen.py photo.jpg --rectify --show
#writes reader/tests/output/seen.png
import argparse
import sys
from pathlib import Path

import cv2

repoFolder = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repoFolder))

from reader import glyphs, rectify, score, staves

#colours are blue, green, red
colours = {
    "blackHead": (0, 160, 0), "halfHead": (0, 160, 0), "wholeHead": (0, 160, 0),
    "flag8Up": (200, 0, 200), "flag8Down": (200, 0, 200), "flag16Up": (200, 0, 200), "flag16Down": (200, 0, 200),
    "sharp": (0, 140, 255), "flat": (0, 140, 255), "natural": (0, 140, 255),
    "gClef": (120, 120, 120), "fClef": (120, 120, 120), "four": (120, 120, 120),
}
restColour = (255, 120, 0)
dotColour = (0, 200, 255)


def drawUsed(source, gray, picture, spacing):
    #run the real reader, catching every note it accepts as it times the four voices
    kept = []
    original = score.chooseBar

    def recording(*arguments):
        timed, problems = original(*arguments)
        kept.extend(timed)
        return timed, problems

    score.chooseBar = recording
    try:
        score.readScore(source)
        print("the reader read every bar")
    except score.ReaderError as error:
        print("the reader stopped: " + str(error))
    finally:
        score.chooseBar = original
    voiceColours = {"soprano": (0, 0, 255), "alto": (0, 150, 0), "tenor": (255, 0, 0), "bass": (180, 0, 180)}
    for onset, event, voiceName in kept:
        colour = voiceColours[voiceName]
        left, top = int(event.x - event.width / 2), int(event.y - spacing / 2)
        cv2.rectangle(picture, (left, top), (left + int(event.width), top + int(spacing)), colour, 2)
        label = f"{voiceName[0].upper()} {event.beats:g} ({event.confidence:.2f})"
        cv2.putText(picture, label, (left, top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
    print(f"{len(kept)} notes kept. key: red soprano, green alto, blue tenor, purple bass; label is voice, beats, confidence")


def main():
    parser = argparse.ArgumentParser(description="draw what the reader sees")
    parser.add_argument("image")
    parser.add_argument("--rectify", action="store_true", help="clean up a photo first, as python -m reader --rectify does")
    parser.add_argument("--show", action="store_true", help="pop the picture up as well")
    parser.add_argument("--used", action="store_true", help="only draw the notes the reader actually kept, with their voice")
    arguments = parser.parse_args()

    source = arguments.image
    outputFolder = Path(__file__).resolve().parent / "output"
    outputFolder.mkdir(exist_ok=True)
    if arguments.rectify:
        source = str(outputFolder / "seen.clean.png")
        cv2.imwrite(source, rectify.rectify(arguments.image))

    gray, found, spacing = score.normalise(staves.loadGray(source))
    matches, dots = glyphs.findGlyphs(staves.eraseLines(gray, found, spacing), spacing)
    picture = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for staff in found:
        for y in staff.lines:
            cv2.line(picture, (int(staff.left), int(y)), (int(staff.right), int(y)), (255, 200, 0), 1)
    for index in range(0, len(found) - 1, 2):
        try:
            barlines = staves.findBarlines(gray < staves.lightLevel, found[index], found[index + 1], spacing)
        except Exception:
            barlines = []
        for x in barlines:
            cv2.line(picture, (int(x), int(found[index].lines[0] - spacing)), (int(x), int(found[index + 1].lines[4] + spacing)), (0, 0, 255), 3)

    if arguments.used:
        drawUsed(source, gray, picture, spacing)
        matches, dots = [], []
    for match in matches:
        colour = restColour if match.name in glyphs.restNames else colours.get(match.name, (0, 0, 0))
        left, top = int(match.cx - match.width / 2), int(match.cy - match.height / 2)
        cv2.rectangle(picture, (left, top), (left + int(match.width), top + int(match.height)), colour, 2)
        label = f"{match.name.replace('Head', '')} {match.score:.2f}"
        cv2.putText(picture, label, (left, top - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
    for dot in dots:
        cv2.circle(picture, (int(dot.cx), int(dot.cy)), int(spacing / 3), dotColour, 2)

    outputPath = outputFolder / "seen.png"
    cv2.imwrite(str(outputPath), picture)
    print(f"{len(found)} staves, {len(matches)} symbols, {len(dots)} dots, staff spacing {spacing:.1f}px -> {outputPath}")
    print("key: light blue staff lines, red barlines, green notes, purple flags, orange rests,")
    print("     yellow circles dots, blue-orange accidentals, grey clefs and time signature; numbers are match scores")
    if arguments.show:
        cv2.imshow("what the reader sees - any key closes", picture)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


main()
