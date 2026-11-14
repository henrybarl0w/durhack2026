import random
import sys
from pathlib import Path

import mido

testsFolder = Path(__file__).resolve().parent
sys.path.insert(0, str(testsFolder.parent.parent))

from stubEncoding import textToMidi

midiFolder = testsFolder / "midi"
ticksPerBeat = 480


def buildTrack(items):
    #items are (pitch or None, beats) one after another, a list of pitches is a chord
    track = mido.MidiTrack()
    pendingRest = 0
    for pitch, beats in items:
        ticks = round(beats * ticksPerBeat)
        if pitch is None:
            pendingRest += ticks
            continue
        pitches = pitch if isinstance(pitch, list) else [pitch]
        for index, note in enumerate(pitches):
            track.append(mido.Message("note_on", note=note, velocity=80, time=pendingRest if index == 0 else 0))
        for index, note in enumerate(pitches):
            track.append(mido.Message("note_off", note=note, velocity=0, time=ticks if index == 0 else 0))
        pendingRest = 0
    return track


def saveMidi(name, tracks):
    midiFile = mido.MidiFile(ticks_per_beat=ticksPerBeat)
    midiFile.tracks.extend(tracks)
    midiFile.save(midiFolder / name)


def main():
    midiFolder.mkdir(exist_ok=True)

    saveMidi("scale.mid", [buildTrack([(pitch, 1) for pitch in (60, 62, 64, 65, 67, 69, 71, 72)])])

    saveMidi("chromatic.mid", [buildTrack([(pitch, 0.5) for pitch in range(48, 73)])])

    twinkle = [60, 60, 67, 67, 69, 69, 67, 65, 65, 64, 64, 62, 62, 60]
    twinkleBeats = [1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2]
    saveMidi("twinkle.mid", [buildTrack(list(zip(twinkle, twinkleBeats)))])

    arpeggio = [36, 40, 43, 48, 52, 55, 60, 64, 67, 72, 76, 79, 84]
    saveMidi("arpeggio.mid", [buildTrack([(pitch, 0.5) for pitch in arpeggio + arpeggio[-2::-1]])])

    saveMidi("rhythms.mid", [buildTrack([
        (60, 4), (62, 2), (64, 1.5), (65, 0.5), (67, 0.25), (69, 0.25), (71, 0.5),
        (None, 1), (72, 0.5), (None, 0.5), (71, 3), (69, 2.5), (67, 1), (65, 0.75), (64, 0.25), (62, 4),
    ])])

    saveMidi("chords.mid", [buildTrack([
        ([48, 52, 55], 2), ([53, 57, 60], 2), ([55, 59, 62], 2), ([48, 52, 55, 60], 4),
    ])])

    melody = buildTrack([(67, 1), (69, 1), (71, 1), (72, 1), (74, 2), (72, 2)])
    bass = buildTrack([(43, 2), (45, 2), (47, 2), (48, 2)])
    saveMidi("twoTracks.mid", [melody, bass])

    generator = random.Random(1)
    longPitches = [generator.randint(40, 84) for _ in range(200)]
    longBeats = [generator.choice([0.5, 1, 1, 1.5, 2]) for _ in range(200)]
    saveMidi("longRandom.mid", [buildTrack(list(zip(longPitches, longBeats)))])

    saveMidi("empty.mid", [mido.MidiTrack()])

    (midiFolder / "stubEncoded.mid").write_bytes(textToMidi("Hello, world! café"))

    print("wrote", len(list(midiFolder.glob("*.mid"))), "files to", midiFolder)


main()
