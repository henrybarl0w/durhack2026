import io
import os
from collections import namedtuple

import mido

from .errors import RendererError

Note = namedtuple("Note", ["pitch", "start", "end"])


def loadMidi(source):
    try:
        if isinstance(source, (bytes, bytearray)):
            return mido.MidiFile(file=io.BytesIO(bytes(source)))
        return mido.MidiFile(os.fspath(source))
    except FileNotFoundError:
        raise RendererError("midi file not found: " + str(source))
    except Exception as error:
        raise RendererError("invalid midi file: " + str(error))


def readNotes(source):
    midiFile = loadMidi(source)
    if midiFile.ticks_per_beat <= 0:
        raise RendererError("unsupported midi timing")
    notes = []
    openNotes = {}
    tick = 0
    for message in mido.merge_tracks(midiFile.tracks):
        tick += message.time
        if message.type not in ("note_on", "note_off"):
            continue
        isStart = message.type == "note_on" and message.velocity > 0
        if isStart:
            openNotes.setdefault(message.note, tick)
        elif message.note in openNotes:
            notes.append(Note(message.note, openNotes.pop(message.note), tick))
    notes.sort(key=lambda note: note.start)
    return notes, midiFile.ticks_per_beat


def writeMidi(notes, ticksPerBeat):
    midiFile = mido.MidiFile(ticks_per_beat=ticksPerBeat)
    track = mido.MidiTrack()
    midiFile.tracks.append(track)
    events = []
    for note in map(lambda item: Note(*item), notes):
        events.append((note.start, 1, mido.Message("note_on", note=note.pitch, velocity=64)))
        events.append((note.end, 0, mido.Message("note_off", note=note.pitch, velocity=0)))
    previousTick = 0
    for tick, _, message in sorted(events, key=lambda event: event[:2]):
        track.append(message.copy(time=tick - previousTick))
        previousTick = tick
    buffer = io.BytesIO()
    midiFile.save(file=buffer)
    return buffer.getvalue()
