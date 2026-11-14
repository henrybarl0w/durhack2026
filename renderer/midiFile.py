import io
import os
from collections import namedtuple

import mido

from .errors import RendererError

Note = namedtuple("Note", ["pitch", "start", "end", "voice"], defaults=[None])
Metadata = namedtuple("Metadata", ["timeSignature", "key", "tempo"], defaults=[None, None, None])

drumChannel = 9


def loadMidi(source):
    try:
        if isinstance(source, (bytes, bytearray)):
            return mido.MidiFile(file=io.BytesIO(bytes(source)))
        return mido.MidiFile(os.fspath(source))
    except FileNotFoundError:
        raise RendererError("midi file not found: " + str(source))
    except Exception as error:
        raise RendererError("invalid midi file: " + str(error))


def readPiece(source):
    midiFile = loadMidi(source)
    if midiFile.ticks_per_beat <= 0:
        raise RendererError("unsupported midi timing")
    found = []
    metadata = {}
    for trackIndex, track in enumerate(midiFile.tracks):
        openNotes = {}
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "time_signature":
                metadata.setdefault("timeSignature", (message.numerator, message.denominator))
            elif message.type == "key_signature":
                metadata.setdefault("key", message.key)
            elif message.type == "set_tempo":
                metadata.setdefault("tempo", round(mido.tempo2bpm(message.tempo)))
            if message.type not in ("note_on", "note_off") or message.channel == drumChannel:
                continue
            key = (message.channel, message.note)
            if message.type == "note_on" and message.velocity > 0:
                openNotes.setdefault(key, tick)
            elif key in openNotes:
                found.append(((trackIndex, message.channel), message.note, openNotes.pop(key), tick))
    voiceKeys = sorted({voiceKey for voiceKey, _, _, _ in found})
    voiceNumbers = {voiceKey: number for number, voiceKey in enumerate(voiceKeys)} if len(voiceKeys) > 1 else {}
    notes = [Note(pitch, start, end, voiceNumbers.get(voiceKey)) for voiceKey, pitch, start, end in found]
    notes.sort(key=lambda note: note.start)
    return notes, midiFile.ticks_per_beat, Metadata(**metadata)


def readNotes(source):
    notes, ticksPerBeat, _ = readPiece(source)
    return notes, ticksPerBeat


def writeMidi(notes, ticksPerBeat):
    midiFile = mido.MidiFile(ticks_per_beat=ticksPerBeat)
    notesByVoice = {}
    for note in map(lambda item: Note(*item), notes):
        notesByVoice.setdefault(note.voice or 0, []).append(note)
    for voice in sorted(notesByVoice or {0: []}):
        events = []
        for note in notesByVoice.get(voice, []):
            events.append((note.start, 1, mido.Message("note_on", note=note.pitch, velocity=64)))
            events.append((note.end, 0, mido.Message("note_off", note=note.pitch, velocity=0)))
        track = mido.MidiTrack()
        previousTick = 0
        for tick, _, message in sorted(events, key=lambda event: event[:2]):
            track.append(message.copy(time=tick - previousTick))
            previousTick = tick
        midiFile.tracks.append(track)
    buffer = io.BytesIO()
    midiFile.save(file=buffer)
    return buffer.getvalue()
