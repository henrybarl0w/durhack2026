#stub: throwaway text to midi encoder so the tests have something to feed the renderer

import renderer

ticksPerBeat = 480


def textToMidi(text):
    notes = []
    tick = 0
    for byteValue in text.encode("utf-8"):
        for nibble in (byteValue >> 4, byteValue & 15):
            notes.append(renderer.Note(48 + nibble, tick, tick + ticksPerBeat))
            tick += ticksPerBeat
    return renderer.writeMidi(notes, ticksPerBeat)
