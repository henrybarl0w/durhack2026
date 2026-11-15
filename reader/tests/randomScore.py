#makes a random, very plain four-part chorale: a .mid and its rendered .png, standing in for the encoder
#the limits it keeps are the ones the reader relies on: 4/4, four voices (soprano and alto on the top staff,
#tenor and bass below), halves, quarters and eighths that never cross a barline (so no ties), no rests,
#no two voices on the same note, no voice crossing, each voice in its own range, one random major key
#run from the repo folder:
#   python reader/tests/randomScore.py
#   python reader/tests/randomScore.py --seed 7 --bars 8 --key Eb --output myScore --show
import argparse
import random
import sys
from pathlib import Path

repoFolder = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repoFolder))

import renderer

ticksPerBeat = 480
#soprano, alto, tenor, bass
voiceRanges = [(60, 79), (55, 72), (48, 67), (40, 60)]
keyTonics = {"C": 0, "G": 7, "D": 2, "A": 9, "E": 4, "F": 5, "Bb": 10, "Eb": 3, "Ab": 8}
majorScale = (0, 2, 4, 5, 7, 9, 11)
#chords as scale degrees (0 is the tonic), and which chords may follow each
chordTones = {0: (0, 2, 4), 1: (1, 3, 5), 3: (3, 5, 0), 4: (4, 6, 1), 5: (5, 0, 2)}
nextChords = {0: [3, 4, 5, 1], 1: [4], 3: [4, 0, 1], 4: [0, 5], 5: [3, 1, 4]}
#how a bar is split into chords, in beats
barRhythms = [[1, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1], [1, 1, 2], [2, 2]]


def scalePitches(tonic, low, high, degrees=None):
    return [pitch for pitch in range(low, high + 1)
            if (pitch - tonic) % 12 in [majorScale[degree] for degree in (degrees or range(7))]]


def voiceChord(generator, tonic, chord, previous):
    #pick one chord tone per voice, top to bottom, strictly falling, staying near the last chord
    for _ in range(200):
        voicing = []
        for voice, (low, high) in enumerate(voiceRanges):
            ceiling = min(high, voicing[-1] - 1) if voicing else high
            options = scalePitches(tonic, low, ceiling, chordTones[chord])
            if previous:
                options = [pitch for pitch in options if abs(pitch - previous[voice]) <= 7] or options
            if not options:
                break
            voicing.append(generator.choice(options))
        if len(voicing) == 4 and len({pitch % 12 for pitch in voicing}) == 3:
            return voicing
    return None


def chorale(generator, bars, tonic):
    #returns (pitch, startTick, endTick, voice)
    notes = []
    chord, previous, beat = 0, None, 0.0
    for bar in range(bars):
        rhythm = [4] if bar == bars - 1 else generator.choice(barRhythms)
        for index, length in enumerate(rhythm):
            if bar == bars - 1:
                chord = 0
            elif bar == bars - 2 and index == len(rhythm) - 1:
                chord = 4
            voicing = voiceChord(generator, tonic, chord, previous)
            while voicing is None:
                chord = generator.choice(list(chordTones))
                voicing = voiceChord(generator, tonic, chord, previous)
            for voice, pitch in enumerate(voicing):
                start = round(beat * ticksPerBeat)
                end = round((beat + length) * ticksPerBeat)
                #now and then one voice moves in two eighths: a step to a passing note on the off-beat
                isPassing = length == 1 and voice in (1, 2) and generator.random() < 0.2
                passing = []
                if isPassing:
                    below, above = voicing[voice + 1], voicing[voice - 1]
                    step = scalePitches(tonic, pitch - 2, pitch + 2)
                    passing = [other for other in step if other != pitch and below < other < above]
                if passing:
                    middle = start + ticksPerBeat // 2
                    notes.append((pitch, start, middle, voice))
                    notes.append((generator.choice(passing), middle, end, voice))
                else:
                    notes.append((pitch, start, end, voice))
            previous, beat = voicing, beat + length
            chord = generator.choice(nextChords[chord])
    return notes


def main():
    parser = argparse.ArgumentParser(description="write a random plain chorale as .mid and .png")
    parser.add_argument("--seed", type=int, help="same seed, same score (default: random)")
    parser.add_argument("--bars", type=int, default=6)
    parser.add_argument("--key", choices=sorted(keyTonics), help="default: random")
    parser.add_argument("--width", type=int, default=2000)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "output" / "random"),
                        help="path without extension, .mid and .png are added")
    parser.add_argument("--show", action="store_true", help="pop up the score, any key closes the window")
    arguments = parser.parse_args()

    seed = arguments.seed if arguments.seed is not None else random.randrange(1_000_000)
    generator = random.Random(seed)
    key = arguments.key or generator.choice(sorted(keyTonics))
    notes = chorale(generator, max(2, arguments.bars), keyTonics[key])
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    midiPath, pngPath = output.with_suffix(".mid"), output.with_suffix(".png")
    midiPath.write_bytes(renderer.writeMidi(notes, ticksPerBeat))
    renderer.midiToScore(midiPath, pngPath, width=arguments.width, zoom=arguments.zoom, key=key)
    print(f"seed {seed}, key {key}: {len(notes)} notes -> {midiPath} and {pngPath}")
    if arguments.show:
        show(pngPath, f"random chorale, seed {seed}, {key} major - any key closes")


def show(imagePath, title, maxWidth=1600, maxHeight=900):
    #fits the score to the screen; press any key or close the window to carry on
    import cv2

    image = cv2.imread(str(imagePath))
    scale = min(1.0, maxWidth / image.shape[1], maxHeight / image.shape[0])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, image.shape[1], image.shape[0])
    cv2.imshow(title, image)
    while cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
