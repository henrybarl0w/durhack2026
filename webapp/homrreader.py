#reads a page with homr (a trained music-recognition model) instead of our own reader, and hands back the same thing
#our reader does: notes as (pitch, start tick, end tick, confidence) at 480 ticks per beat. homr writes MusicXML, so the
#page really is turned into an xml first and the notes are taken from that, exactly as they are.
#homr lives in its own virtual environment, webapp/.homr, so its packages never mix with the web app's:
#   python -m venv webapp/.homr
#   webapp/.homr/Scripts/python -m pip install homr onnxruntime      (bin/ instead of Scripts/ outside Windows)
#set HOMR_EXE to use another install. its model files download the first time it runs.
import os
import shutil
import subprocess
import sys
import tempfile

from collections import namedtuple

from reader import Note, ReaderError


class StaffNote(namedtuple("StaffNote", Note._fields + ("staff",))):
    #a Note that also says which staff it was written on (0 treble, 1 bass). the decoder uses it to tell which of
    #two notes at the same pitch belongs to which voices: a pitch doubled across the two staves is two notes, not one
    __slots__ = ()

ticksPerBeat = 480
here = os.path.dirname(os.path.abspath(__file__))
exeName = "homr.exe" if sys.platform == "win32" else "homr"
homrPath = os.environ.get("HOMR_EXE") or os.path.join(here, ".homr", "Scripts" if sys.platform == "win32" else "bin", exeName)
#a page takes homr a good while on a plain computer
timeoutSeconds = 600


def available():
    return os.path.exists(homrPath)


def notesFromXml(xmlPath):
    #every pitched note of the MusicXML, chords split into their notes, rests dropped. each note is timed from the start
    #of its OWN bar (bar number times four beats, plus how far into the bar it sits), not from the start of the piece:
    #one wrong note length in one bar would otherwise push every later note out of time. bar 1 is time zero, the same
    #zero the decoder uses for the scan
    import music21

    score = music21.converter.parse(xmlPath)
    found = []
    for staff, part in enumerate(score.parts):
        for measure in part.getElementsByClass("Measure"):
            for element in measure.recurse().notes:
                inside = float(element.getOffsetInHierarchy(measure))
                for pitch in element.pitches:
                    found.append((measure.number, pitch.midi, inside, inside + float(element.quarterLength), min(staff, 1)))
    if not found:
        raise ReaderError("homr found no notes on the page")
    first = min(number for number, _, _, _, _ in found)
    notes = {}
    for number, pitch, start, end, staff in found:
        begin = (number - first) * 4 + start
        key = (pitch, round(begin * 4), staff)
        if key not in notes:
            notes[key] = StaffNote(pitch, round(begin * ticksPerBeat), round((begin + end - start) * ticksPerBeat), 1.0, staff)
    return sorted(notes.values(), key=lambda note: (note.start, note.pitch))


def readPage(imagePath, keepFolder=None):
    #(notes, ticksPerBeat, path of the MusicXML). the xml is copied to keepFolder if given (the debug dump keeps it)
    work = tempfile.mkdtemp(prefix="homr-")
    try:
        page = os.path.join(work, "page.png")
        shutil.copyfile(imagePath, page)
        try:
            done = subprocess.run([homrPath, page], cwd=work, capture_output=True, text=True, timeout=timeoutSeconds)
        except subprocess.TimeoutExpired:
            raise ReaderError(f"homr took longer than {timeoutSeconds} seconds")
        xml = os.path.join(work, "page.musicxml")
        if not os.path.exists(xml):
            tail = (done.stderr or done.stdout or "").strip().splitlines()[-1:] or ["no output"]
            raise ReaderError("homr could not read the page: " + tail[0])
        kept = xml
        if keepFolder:
            os.makedirs(keepFolder, exist_ok=True)
            kept = os.path.join(keepFolder, "homr.musicxml")
            shutil.copyfile(xml, kept)
        return notesFromXml(xml), ticksPerBeat, kept
    finally:
        shutil.rmtree(work, ignore_errors=True)
