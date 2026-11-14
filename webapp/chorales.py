#text <-> chorale: the web app's encoder and decoder around chorale_engine, renderer and reader
#   text -> integer -> chords hidden in a Bach chorale -> musicxml -> png
#   notes read from a scan -> chords -> (repaired against the engine's legal options) -> integer -> text
import base64
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from importlib.resources import files

import chorale_engine
import renderer
import verovio

capacityFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chorale_capacity.json")
#the integer is the message bytes behind one marker byte, so a message with leading zero bytes survives
marker = b"\x01"
#a chorale is only named if this much of its soprano line is found in the scan
minSopranoMatch = 0.5

capacities = None
tunes = {}


class ChoraleError(Exception):
    pass


def loadCapacities():
    #{chorale id: (capacity, beats)} for every usable chorale, worked out once and kept in a file
    global capacities
    if capacities is not None:
        return capacities
    saved = {}
    if os.path.exists(capacityFile):
        with open(capacityFile) as file:
            saved = json.load(file)
    changed = False
    for chorale in chorale_engine.usable_chorales():
        if str(chorale) not in saved:
            capacity = chorale_engine.capacity_report(chorale)["capacity"]
            saved[str(chorale)] = [str(capacity), len(chorale_engine.load_tune(chorale).beats)]
            changed = True
    if changed:
        with open(capacityFile, "w") as file:
            json.dump(saved, file)
    #capacities are huge integers, kept as text in the file
    capacities = {int(chorale): (int(capacity), beats) for chorale, (capacity, beats) in saved.items()}
    return capacities


def loadTune(chorale):
    if chorale not in tunes:
        tunes[chorale] = chorale_engine.load_tune(chorale)
    return tunes[chorale]


def textToInt(text):
    return int.from_bytes(marker + text.encode("utf-8"), "big")


def intToText(number):
    data = number.to_bytes((number.bit_length() + 7) // 8, "big")
    if not data.startswith(marker):
        raise ChoraleError("this does not decode to a message")
    try:
        return data[1:].decode("utf-8")
    except UnicodeDecodeError:
        raise ChoraleError("this decodes to garbage, so some notes were misread")


def maxBytes():
    #the most message bytes the biggest chorale holds
    biggest = max(capacity for capacity, _ in loadCapacities().values())
    return (biggest.bit_length() - 1) // 8 - 1


def encodeText(text):
    #returns (png bytes, chorale id); the shortest chorale with room for the message
    if len(text.encode("utf-8")) > 256:
        raise ChoraleError(f"message too long: at most about {maxBytes()} bytes fit in one chorale")
    number = textToInt(text)
    fits = [(beats, chorale) for chorale, (capacity, beats) in loadCapacities().items() if number < capacity]
    if not fits:
        raise ChoraleError(f"message too long: at most about {maxBytes()} bytes fit in one chorale")
    #a few chorales may not draw (verovio rejects the score): fall through to the next shortest that fits
    failed = []
    for _, chorale in sorted(fits):
        try:
            return drawChorale(chorale, number), chorale
        except renderer.RendererError as error:
            failed.append(f"{chorale}: {error}")
    raise ChoraleError("could not draw any chorale that fits (" + "; ".join(failed[:3]) + ")")


def drawChorale(chorale, number):
    #chords -> notes -> a grand staff (treble and bass, two voices on each): half the page of four separate staves,
    #and the layout the reader reads, so what is drawn here is what is scanned back
    chords = chorale_engine.encode(chorale, number)
    tune = loadTune(chorale)
    ticks = 480
    notes = []

    def span(offset, length):
        #snap to a sixteenth-note grid, never shorter than one sixteenth: odd lengths make the renderer write bad xml
        start = round(offset * 4) * (ticks // 4)
        return start, max(start + ticks // 4, round((offset + length) * 4) * (ticks // 4))

    #the soprano keeps the tune's own rhythm; alto, tenor and bass hold one note per beat. voices 0-3 are S A T B
    for event in tune.soprano:
        if event.midi is not None:
            notes.append((event.midi, *span(event.offset, event.length), 0))
    for beat, chord in zip(tune.beats, chords):
        for voice in (1, 2, 3):
            notes.append((chord[voice], *span(beat.offset, beat.length), voice))
    #the engine names flats like B-; the renderer wants Bb, and m for minor
    key = tune.home_key.tonic.replace("-", "b") + ("m" if tune.home_key.mode == "minor" else "")
    #set verovio's font folder again right before drawing, in case the web server's thread lost it
    verovio.setDefaultResourcePath(str(files("verovio") / "data"))
    return renderer.notesToScore(notes, ticks, None, "png", 2000, 1.0, (4, 4), key)


def pngToDataUri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def soundingAt(notes, tick):
    #pitches of the notes sounding at a tick; a few ticks of slack for notes starting just late
    return {note.pitch for note in notes if note.start <= tick + 5 and note.end > tick + 5}


def readChords(chorale, notes, ticksPerBeat):
    #what the scan shows at each of this chorale's beats
    beats = loadTune(chorale).beats
    first = beats[0].offset
    return [soundingAt(notes, int((beat.offset - first) * ticksPerBeat)) for beat in beats]


def findChorale(notes, ticksPerBeat):
    #the tune is fixed, so the chorale is the one whose soprano line best matches the scan at each beat.
    #pitch names are compared, not exact pitches, so a misread octave still counts
    best, bestScore = None, 0
    for chorale in loadCapacities():
        beats = loadTune(chorale).beats
        chords = readChords(chorale, notes, ticksPerBeat)
        hits = sum(1 for beat, pitches in zip(beats, chords) if beat.soprano_midi % 12 in {pitch % 12 for pitch in pitches})
        score = hits / len(beats)
        if score > bestScore:
            best, bestScore = chorale, score
    if best is None or bestScore < minSopranoMatch:
        raise ChoraleError(f"could not tell which chorale this is (closest was chorale {best}, "
                           f"{round(100 * bestScore)}% of its soprano notes matched, {len(notes)} notes read)")
    return best, bestScore


def mismatch(option, seen):
    #how badly a legal chord fits what was seen: a pitch seen that is not in the chord is a misread (costly),
    #a chord pitch not seen may just be a note the reader missed (cheap)
    return 1.0 * len(seen - set(option)) + 0.4 * len(set(option) - seen)


def bestPath(chorale, phrase, seen):
    #the legal run of chords through one phrase that fits the scan best overall. choosing the nearest chord one
    #beat at a time lets one misread chord poison every option after it, so every legal path is weighed together
    previous, beforeBass = chorale_engine.start_state(chorale, phrase)
    states = {(previous, beforeBass): (0.0, [])}
    for beat in range(len(seen)):
        following = {}
        for (before, beforeBefore), (cost, path) in states.items():
            for option in chorale_engine.options(chorale, phrase, beat, before, beforeBefore):
                total = cost + mismatch(option, seen[beat])
                key = (tuple(option), before[3] if before is not None else None)
                if key not in following or total < following[key][0]:
                    following[key] = (total, path + [tuple(option)])
        #keep the search small
        states = dict(sorted(following.items(), key=lambda item: item[1][0])[:400])
    return min(states.values(), key=lambda item: item[0])[1]


def decodeNotes(notes, ticksPerBeat):
    #returns a dict: text, chorale, fixed (chords that had to be snapped to a legal one), total (chords)
    chorale, match = findChorale(notes, ticksPerBeat)
    tune = loadTune(chorale)
    seen = readChords(chorale, notes, ticksPerBeat)
    number, scale, fixed = 0, 1, 0
    for phrase, (start, end) in enumerate(tune.phrases):
        chords = bestPath(chorale, phrase, seen[start:end])
        fixed += sum(1 for chord, scanned in zip(chords, seen[start:end]) if sorted(set(chord)) != sorted(scanned))
        number += chorale_engine.decode_phrase(chorale, phrase, chords) * scale
        scale *= chorale_engine.phrase_capacity(chorale, phrase)
    return {"text": intToText(number), "chorale": chorale, "fixed": fixed, "total": len(tune.beats),
            "sopranoMatch": round(match, 2)}
