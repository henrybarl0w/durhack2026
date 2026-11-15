#live camera scanning: look for the page's markers in each frame; the moment all eight are in view, that frame is
#the snapshot. it is straightened by the markers, cleaned and read in the background (the phone is told "reading" meanwhile)
import os
import re
import subprocess
import sys
import tempfile
import threading
import warnings

import cv2
import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader import ReaderError, notesToMidi, readScore
from reader.vote import combine, readVoted
from debugdump import saveScan
from chorales import decodeReads, flag
from reader.rectify import cleanPage, findMarkersInGray, resizeLongSide, warpByMarkers

#open a picture of what the reader sees on the computer running the server after every scan
#(off: it runs the reader a second time on the page, which only slows the real read; the debug dump has the picture)
popUpSeen = False
#frames are looked at about this size for the cheap marker check
checkSide = 1200
#a page has eight markers (four corners, four halfway along the edges) and the more of them the reader sees, the
#flatter it can make the page. the snapshot waits for all eight; if only some are in view (a thumb over one) for this
#many frames in a row, it takes what it has, as long as the three corners it needs are there
partialFramesToWait = 8
#photos read: one. the page is read once and the result shown, with anything the reader was unsure of flagged,
#rather than silently scanning again as if something had gone wrong
maxReads = 1

pitchNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pitchName(pitch):
    return pitchNames[pitch % 12] + str(pitch // 12 - 1)


def showWhatReaderSees(imagePath):
    #draws the cleaned page with the reader's staff lines, barlines and every note it kept (reader/tests/showSeen.py)
    #and opens it on the computer running the server. in the background, so the phone is not kept waiting
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(repo, "reader", "tests", "showSeen.py")
    seen = os.path.join(repo, "reader", "tests", "output", "seen.png")

    def run():
        try:
            if os.path.exists(seen):
                os.remove(seen)
            subprocess.run([sys.executable, script, imagePath, "--used"], cwd=repo, capture_output=True, timeout=120)
        except Exception as error:
            print("could not draw what the reader sees:", error, flush=True)
        #if the reader could not even find the staves there is no drawing: show the cleaned page instead
        target = seen if os.path.exists(seen) else imagePath
        if hasattr(os, "startfile"):
            os.startfile(target)
        else:
            print("what the reader saw:", target, flush=True)

    threading.Thread(target=run, daemon=True).start()


class Scanner:
    def __init__(self):
        self.lock = threading.Lock()
        self.generation = 0
        self.reset()

    def reset(self):
        with self.lock:
            #a read still running when this is called belongs to the old generation and its result is dropped
            self.generation += 1
            self.reading = False
            self.outcome = None
            self.corners = None
        self.reads = []
        self.partialFrames = 0
        self.attempts = 0
        self.result = None

    def addFrame(self, jpegBytes):
        with self.lock:
            if self.reading:
                return {"status": "reading the score...", "found": True, "reading": True, "corners": self.corners}
            if self.outcome is not None:
                outcome, self.outcome = self.outcome, None
                return outcome
        photo = cv2.imdecode(numpy.frombuffer(jpegBytes, numpy.uint8), cv2.IMREAD_COLOR)
        if photo is None:
            return {"status": "bad frame"}
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        small = resizeLongSide(gray, checkSide)
        markers = findMarkersInGray(small)
        if markers is None:
            self.partialFrames = 0
            return {"status": "markers not found", "found": False}
        if markers.count() < 8:
            self.partialFrames += 1
            if self.partialFrames < partialFramesToWait:
                return {"status": f"hold still: {markers.count()} of 8 markers in view", "found": False}
        self.partialFrames = 0
        #the page is in view: this frame is the snapshot
        corners = [list(map(float, point / small.shape[1])) for point in (markers.topLeft, markers.topRight, markers.bottomLeft)]
        factor = max(gray.shape) / max(small.shape)
        with self.lock:
            if self.reading:
                return {"status": "reading the score...", "found": True, "reading": True, "corners": self.corners}
            self.reading, self.corners, generation = True, corners, self.generation
        threading.Thread(target=self.work, args=(photo, markers, factor, corners, generation), daemon=True).start()
        return {"status": "markers found, reading the score...", "found": True, "reading": True, "corners": corners}

    def work(self, photo, markers, factor, corners, generation):
        #the read, on its own thread
        try:
            result = self.readPhoto(photo, markers, factor)
        except Exception as error:
            print("scan failed:", repr(error), flush=True)
            result = {"decoded": {"text": None, "chorale": None, "reads": 0, "fixed": 0, "total": 0,
                                  "flags": [flag("reader", f"the scan failed: {error}")], "flagCounts": {"reader": 1}},
                      "found": True, "reads": 0, "notes": [], "ticksPerBeat": 480, "warnings": [], "status": "read", "done": True}
        result["corners"] = corners
        with self.lock:
            if generation == self.generation:
                self.outcome, self.reading = result, False

    def readPhoto(self, photo, markers, factor):
        self.attempts += 1
        #the snapshot, exactly as the camera gave it (kept for the debug dump)
        path = os.path.join(tempfile.gettempdir(), f"durhackScan{self.attempts}.png")
        cv2.imwrite(path, photo)
        readPath = path
        #show what the reader sees after every read, so a bad one can be looked at straight away
        if popUpSeen:
            showWhatReaderSees(path)
        notes, ticksPerBeat, readError, newReads = [], 480, None, []
        trace = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                #the reader needs the page straightened upright by its markers and cleaned first; it then reads it
                #(at three working sizes with the votes taken, if the "vote" switch in reader/features.py is on)
                gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
                warped = warpByMarkers(gray, markers.scaled(factor))
                steps = {}
                cleanPage(warped, True, steps, flattened=markers.count() == 8)
                readPath = os.path.join(tempfile.gettempdir(), f"durhackScan{self.attempts}.clean.png")
                cv2.imwrite(readPath, steps["result"])
                newReads = readVoted(readPath, trace)
                notes, ticksPerBeat = combine(newReads, trace)
            except ReaderError as error:
                readError = str(error)
        if readError is None:
            self.result = (notes, ticksPerBeat)
            self.reads.extend(newReads)
        #notes back to text, from every read so far: the chorale is identified, the reads are combined by voting,
        #misread chords are corrected to the best legal run, then decoded. problems come back as flags
        if self.reads:
            decoded = decodeReads(self.reads)
            if decoded["text"] is None and decoded["chorale"] is None and len(self.reads) > 1:
                #the page may have changed since the earlier reads: try the newest on its own
                newest = decodeReads(self.reads[-1:])
                if newest["chorale"] is not None:
                    decoded, self.reads = newest, self.reads[-1:]
        else:
            decoded = {"text": None, "chorale": None, "reads": 0, "fixed": 0, "total": 0, "flags": [], "flagCounts": {}}
        if readError is not None:
            decoded["flags"].insert(0, flag("reader", "the reader could not read this picture: " + readError))
            decoded["flagCounts"]["reader"] = 1
        #the debug suite: the scan, the read-back midi drawn again, side by side, and a note by note dump of what went
        #wrong (webapp/debug/latest/). it runs in the background; its per-beat detail is not sent to the phone
        saveScan(readPath, notes, ticksPerBeat, decoded, trace, [str(warning.message) for warning in caught], readError)
        decoded.pop("beatsDetail", None)
        #bars the reader could not fully read are flagged too, with the bar number
        for warning in caught:
            found = re.match(r"bar (\d+): (.*)", str(warning.message))
            decoded["flags"].append(flag("bar", found.group(2) if found else str(warning.message),
                                         bar=int(found.group(1)) if found else None))
            decoded["flagCounts"]["bar"] = decoded["flagCounts"].get("bar", 0) + 1
        payload = {
            "decoded": decoded,
            "found": True,
            "reads": len(self.reads),
            "notes": [
                {"name": pitchName(note.pitch), "pitch": note.pitch, "start": note.start, "end": note.end,
                 "confidence": note.confidence}
                for note in notes
            ],
            "ticksPerBeat": ticksPerBeat,
            "warnings": [str(warning.message) for warning in caught],
        }
        problems = sum(decoded["flagCounts"].values())
        if decoded["text"] is None and self.attempts < maxReads:
            #not decoded yet: keep scanning for more evidence, and show what there is so far
            print(f"read {self.attempts}/{maxReads}: {len(notes)} notes, {problems} problems flagged, scanning again", flush=True)
            payload.update(status=f"read {self.attempts} of up to {maxReads}: {len(notes)} notes, {problems} problems "
                                  f"flagged, not decoded yet", done=False, partial=True)
            return payload
        payload.update(status="read", done=True)
        return payload

    def midi(self):
        if self.result is None:
            return None
        notes, ticksPerBeat = self.result
        return notesToMidi(notes, ticksPerBeat)
