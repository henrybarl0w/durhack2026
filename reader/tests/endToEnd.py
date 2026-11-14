#render a midi file, read the image back, render what was read, and compare
#run from the repo folder:
#   python reader/tests/endToEnd.py renderer/tests/midi/choraleFourTracks.mid
#   python reader/tests/endToEnd.py my.mid --width 1000 --zoom 1.5 --show
#writes into reader/tests/output/: original.png, readBack.mid, readBack.png, sideBySide.png
import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import cv2
import numpy

repoFolder = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repoFolder))

import reader
import renderer

noteNames = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def describe(note, ticksPerBeat):
    pitch, start, end = note
    name = noteNames[pitch % 12] + str(pitch // 12 - 1)
    return f"{name} from beat {start / ticksPerBeat:g} for {(end - start) / ticksPerBeat:g} beats"


def stack(top, bottom):
    width = max(top.shape[1], bottom.shape[1])
    pad = lambda image: numpy.pad(image, ((0, 0), (0, width - image.shape[1])), constant_values=255)
    divider = numpy.full((8, width), 160, numpy.uint8)
    return numpy.vstack([pad(top), divider, pad(bottom)])


def main():
    parser = argparse.ArgumentParser(description="renderer -> image -> reader -> renderer, then compare")
    parser.add_argument("midi", help="a .mid file to start from")
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--show", action="store_true", help="pop up the two scores, any key closes the window")
    arguments = parser.parse_args()

    outputFolder = Path(__file__).resolve().parent / "output"
    outputFolder.mkdir(exist_ok=True)
    originalPng = outputFolder / "original.png"
    readBackMidi = outputFolder / "readBack.mid"
    readBackPng = outputFolder / "readBack.png"

    #1. midi -> score image
    renderer.midiToScore(arguments.midi, originalPng, width=arguments.width, zoom=arguments.zoom)
    original, originalTicks = renderer.readNotes(arguments.midi)

    #2. score image -> notes -> midi
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            notes, ticksPerBeat = reader.readScore(originalPng)
        except reader.ReaderError as error:
            print(f"the reader gave up: {error}\nlook at {originalPng}")
            sys.exit(1)
    for warning in caught:
        print("reader warning:", warning.message)
    readBackMidi.write_bytes(reader.notesToMidi(notes, ticksPerBeat))

    #3. read-back midi -> score image, both scores stacked in one picture
    renderer.midiToScore(readBackMidi, readBackPng, width=arguments.width, zoom=arguments.zoom)
    pictures = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in (originalPng, readBackPng)]
    cv2.imwrite(str(outputFolder / "sideBySide.png"), stack(*pictures))

    #4. compare the notes, timed in beats so different ticks per beat still line up
    def inBeats(items, ticks):
        return Counter((item[0], item[1] * ticksPerBeat // ticks, item[2] * ticksPerBeat // ticks) for item in items)

    expected = inBeats([(note.pitch, note.start, note.end) for note in original], originalTicks)
    got = inBeats([note[:3] for note in notes], ticksPerBeat)
    missing = sorted((expected - got).elements(), key=lambda note: note[1])
    extra = sorted((got - expected).elements(), key=lambda note: note[1])

    print(f"original: {sum(expected.values())} notes, read back: {sum(got.values())} notes")
    if not missing and not extra:
        print("exact round trip")
    for note in missing:
        print("  missed:   ", describe(note, ticksPerBeat))
    for note in extra:
        print("  wrongly read:", describe(note, ticksPerBeat))
    lowest = sorted(notes, key=lambda note: note.confidence)[:3]
    if lowest:
        print("least confident:", ", ".join(f"{describe(note[:3], ticksPerBeat)} ({note.confidence:.2f})" for note in lowest))
    print(f"pictures in {outputFolder} (sideBySide.png: original on top, read back below)")
    if arguments.show:
        show(stack(*pictures))
    sys.exit(0 if not missing and not extra else 1)


def show(image, maxWidth=1600, maxHeight=900):
    #shrinks big scores to fit the screen; press any key or close the window to carry on
    scale = min(1.0, maxWidth / image.shape[1], maxHeight / image.shape[0])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    title = "original (top) vs read back (bottom) - any key closes"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, image.shape[1], image.shape[0])
    cv2.imshow(title, image)
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
    cv2.destroyAllWindows()


main()
