import argparse
import sys
import warnings
from pathlib import Path

import cv2

from .errors import ReaderError
from .midiWriter import notesToMidi
from .rectify import rectify
from .score import readScore


def main():
    parser = argparse.ArgumentParser(prog="reader", description="turn a rendered score image back into midi")
    parser.add_argument("--input", required=True, help="score image, as made by the renderer")
    parser.add_argument("--output", required=True, help="path for the .mid file")
    parser.add_argument("--rectify", action="store_true", help="the input is a photo: straighten and clean it first")
    arguments = parser.parse_args()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            source = arguments.input
            if arguments.rectify:
                source = str(Path(arguments.output).with_suffix(".clean.png"))
                cleaned = rectify(arguments.input)
                cv2.imwrite(source, cleaned)
                print("cleaned photo written to " + source)
            try:
                notes, ticksPerBeat = readScore(source)
            except ReaderError:
                if not arguments.rectify:
                    raise
                #a photo can come out upside down: try it the other way up before giving up
                cv2.imwrite(source, cv2.rotate(cleaned, cv2.ROTATE_180))
                notes, ticksPerBeat = readScore(source)
            with open(arguments.output, "wb") as outputFile:
                outputFile.write(notesToMidi(notes, ticksPerBeat))
        except (ReaderError, OSError, ValueError) as error:
            print("error: " + str(error), file=sys.stderr)
            sys.exit(1)
    for warning in caught:
        print("warning: " + str(warning.message), file=sys.stderr)
    lowest = min((note.confidence for note in notes), default=1.0)
    print(f"wrote {arguments.output} ({len(notes)} notes, lowest confidence {lowest:.2f})")


main()
