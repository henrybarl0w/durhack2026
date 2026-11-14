import struct


def variableLength(value):
    data = [value & 0x7F]
    value >>= 7
    while value:
        data.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(data)


def notesToMidi(notes, ticksPerBeat, velocity=64):
    #a single-track standard midi file, notes are (pitch, startTick, endTick, ...)
    events = []
    for note in notes:
        pitch, start, end = note[0], note[1], note[2]
        events.append((start, 1, bytes([0x90, pitch, velocity])))
        events.append((end, 0, bytes([0x80, pitch, 0])))
    events.sort(key=lambda event: event[:2])
    track = b""
    previous = 0
    for tick, _, message in events:
        track += variableLength(tick - previous) + message
        previous = tick
    track += variableLength(0) + b"\xff\x2f\x00"
    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, ticksPerBeat)
    return header + b"MTrk" + struct.pack(">I", len(track)) + track
