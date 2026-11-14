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


def saveMidi(name, tracks, meta=()):
    midiFile = mido.MidiFile(ticks_per_beat=ticksPerBeat)
    midiFile.tracks.extend(tracks)
    midiFile.tracks[0][:0] = list(meta)
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

    soprano = [(76, 1), (74, 1), (72, 1), (71, 1), (69, 1), (67, 1), (69, 1), (71, 1), (72, 1), (72, 1), (71, 1), (72, 1), (77, 2), (74, 1), (72, 1), (72, 4)]
    alto = [(67, 1), (67, 1), (64, 1), (64, 1), (65, 1), (64, 1), (66, 1), (67, 1), (67, 1), (69, 1), (65, 1), (67, 1), (69, 2), (67, 1), (67, 1), (64, 4)]
    tenor = [(60, 1), (59, 1), (57, 1), (55, 1), (60, 1), (60, 1), (62, 1), (62, 1), (64, 1), (65, 1), (62, 1), (64, 1), (62, 2), (59, 1), (60, 1), (55, 4)]
    bass = [(48, 1), (43, 1), (45, 1), (52, 1), (41, 1), (48, 1), (50, 1), (43, 1), (48, 1), (41, 1), (43, 1), (48, 1), (50, 2), (43, 1), (48, 1), (48, 4)]
    voices = [soprano, alto, tenor, bass]
    saveMidi("choraleFourTracks.mid", [buildTrack(voice) for voice in voices])

    def chordsOf(voiceA, voiceB):
        return [([a[0], b[0]], a[1]) for a, b in zip(voiceA, voiceB)]

    saveMidi("choralePianoTwoTracks.mid", [buildTrack(chordsOf(soprano, alto)), buildTrack(chordsOf(tenor, bass))])
    allChords = [([s[0], a[0], t[0], b[0]], s[1]) for s, a, t, b in zip(soprano, alto, tenor, bass)]
    saveMidi("choraleOneTrackChords.mid", [buildTrack(allChords)])

    def transposed(voice, semitones):
        return [(pitch + semitones, beats) for pitch, beats in voice]

    def metaMessages(numerator, denominator, key, bpm):
        return [
            mido.MetaMessage("time_signature", numerator=numerator, denominator=denominator),
            mido.MetaMessage("key_signature", key=key),
            mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)),
        ]

    saveMidi("choraleFMajor.mid", [buildTrack(transposed(voice, -7)) for voice in voices], metaMessages(4, 4, "F", 72))
    saveMidi("choraleGMajor34.mid", [buildTrack(transposed(voice, -5)) for voice in voices], metaMessages(3, 4, "G", 80))

    aMinor = [(69, 0.5), (71, 0.5), (72, 0.5), (74, 0.5), (76, 0.5), (77, 0.5), (68, 0.5), (69, 1.5), (64, 1)]
    saveMidi("aMinor68.mid", [buildTrack(aMinor)], metaMessages(6, 8, "Am", 100))

    saveMidi("empty.mid", [mido.MidiTrack()])

    (midiFolder / "stubEncoded.mid").write_bytes(textToMidi("Hello, world! café"))

    print("wrote", len(list(midiFolder.glob("*.mid"))), "files to", midiFolder)


main()
