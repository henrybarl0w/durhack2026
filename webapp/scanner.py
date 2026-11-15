#live camera scanning: find the corner markers in each frame, and once they hold steady,
#warp a few frames upright, merge them to wipe out glare, and run the reader on the result
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import warnings

import cv2
import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader import ReaderError, notesToMidi, readScore
from reader.vote import combine, readVoted
import homrreader
from debugdump import saveScan
from chorales import decodeReads, flag
from reader.rectify import cleanPage, findMarkersInGray, resizeLongSide, warpByMarkers

#open a picture of what the reader sees on the computer running the server after every scan
#(off: it runs our own reader a second time on the page, which only competes with homr for the processor; the debug dump has the picture)
popUpSeen = False
#read pages with homr when it is installed (webapp/homrreader.py); our own reader is the fallback
useHomr = True
#frames are looked at about this size for the cheap marker check
checkSide = 1200
#markers in this many frames before collecting starts
steadyNeeded = 2
#frames in a row without markers before starting over
missesAllowed = 4
#merged pictures read: one. the page is read once from the merged frames and the result shown, with anything
#the reader was unsure of flagged, rather than silently scanning again as if something had gone wrong
maxReads = 1
#frames merged into one picture
mergeCount = 4
#a frame whose marker box differs this much (fraction) from the first frame's is a wrong find, and is skipped
shapeTolerance = 0.12
#wrong-looking frames in a row before the first frame is doubted instead
oddAllowed = 4
#a gap longer than this (seconds) starts over
staleAfter = 3.0

pitchNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pitchName(pitch):
    return pitchNames[pitch % 12] + str(pitch // 12 - 1)


def showWhatReaderSees(imagePath):
    #draws the merged picture with the reader's staff lines, barlines and every note it kept (reader/tests/showSeen.py)
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
        #if the reader could not even find the staves there is no drawing: show the merged picture instead
        target = seen if os.path.exists(seen) else imagePath
        if hasattr(os, "startfile"):
            os.startfile(target)
        else:
            print("what the reader saw:", target, flush=True)

    threading.Thread(target=run, daemon=True).start()


def markerShape(markers):
    #two numbers that stay the same however the page is held: how tall the marker box is against its width,
    #and how big a marker is against that width. three wrong squares give different numbers
    topLeft, topRight, bottomLeft, size = markers
    across = numpy.linalg.norm(topRight - topLeft)
    return numpy.linalg.norm(bottomLeft - topLeft) / across, size / across


def sameShape(shape, reference):
    return all(abs(value / wanted - 1) < shapeTolerance for value, wanted in zip(shape, reference))


def alignFrames(frames):
    #the three markers place each frame to within a couple of pixels; shift the others to match the first exactly,
    #or thin staff lines come out doubled and blurred in the merge
    height = min(frame.shape[0] for frame in frames)
    width = min(frame.shape[1] for frame in frames)
    first = frames[0][:height, :width]
    window = cv2.createHanningWindow((width, height), cv2.CV_32F)
    aligned = [first]
    for frame in frames[1:]:
        frame = frame[:height, :width]
        (shiftX, shiftY), _ = cv2.phaseCorrelate(first.astype(numpy.float32), frame.astype(numpy.float32), window)
        if abs(shiftX) < 40 and abs(shiftY) < 40:
            frame = cv2.warpAffine(frame, numpy.float32([[1, 0, -shiftX], [0, 1, -shiftY]]), (width, height),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        aligned.append(frame)
    return aligned


class Scanner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steady = 0
        self.misses = 0
        self.odd = 0
        self.shape = None
        self.reads = []
        self.attempts = 0
        self.lastPath = None
        self.frames = []
        self.lastTime = 0
        self.result = None

    def addFrame(self, jpegBytes):
        now = time.time()
        if now - self.lastTime > staleAfter:
            self.steady, self.frames, self.shape = 0, [], None
        self.lastTime = now
        image = cv2.imdecode(numpy.frombuffer(jpegBytes, numpy.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"status": "bad frame"}
        gray = resizeLongSide(image, checkSide)
        markers = findMarkersInGray(gray)
        if markers is None:
            #one blurry or glary frame is normal: only start over after several misses in a row
            self.misses += 1
            if self.misses >= missesAllowed:
                self.steady, self.frames, self.shape = 0, [], None
            if self.frames or self.steady:
                return {"status": f"lost the markers ({len(self.frames)}/{mergeCount})", "found": False}
            return {"status": "markers not found", "found": False}
        self.misses = 0
        shape = markerShape(markers)
        if self.shape is None:
            self.shape = shape
        elif not sameShape(shape, self.shape):
            #not the same page layout as the frames so far: probably squares found in the music, not the corner markers
            self.odd += 1
            if self.odd >= oddAllowed:
                #so many frames disagree that the first one was probably the wrong one: start again from this one
                self.steady, self.frames, self.shape, self.odd = 0, [], shape, 0
            return {"status": "markers look wrong", "found": False}
        self.odd = 0
        self.steady += 1
        corners = [list(map(float, point / gray.shape[1])) for point in markers[:3]]
        if self.steady < steadyNeeded:
            return {"status": "markers found", "found": True, "corners": corners}
        #warp each frame upright from the full size picture (the small copy was only for finding the markers):
        #markers line the frames up, so merging is per pixel
        factor = max(image.shape) / max(gray.shape)
        bigMarkers = (markers[0] * factor, markers[1] * factor, markers[2] * factor, markers[3] * factor)
        self.frames.append(warpByMarkers(image, bigMarkers))
        if len(self.frames) < mergeCount:
            return {"status": f"capturing ({len(self.frames)}/{mergeCount})", "found": True, "corners": corners}
        frames, self.frames, self.steady = self.frames, [], 0
        result = self.readMerged(frames)
        result["corners"] = corners
        return result

    def readMerged(self, frames):
        stack = numpy.stack(alignFrames(frames))
        #glare brightens, ink darkens: the darkest value of each pixel is ink wherever any frame saw it.
        #the second darkest is a little safer against one frame's motion blur or a shadow
        merged = numpy.sort(stack, axis=0)[min(1, len(frames) - 1)] if len(frames) > 2 else stack.min(axis=0)
        steps = {}
        cleanPage(merged, True, steps)
        self.attempts += 1
        path = os.path.join(tempfile.gettempdir(), f"durhackScan{self.attempts}.png")
        cv2.imwrite(path, steps["result"])
        #show what the reader sees after every read, so a bad one can be looked at straight away
        if popUpSeen:
            showWhatReaderSees(path)
        notes, ticksPerBeat, readError, newReads = [], 480, None, []
        trace = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                done = False
                if useHomr and homrreader.available():
                    #homr turns the page into MusicXML and the notes are taken from that as they are; they then go through
                    #the same repair and decode below as our own reader's did
                    try:
                        notes, ticksPerBeat, xml = homrreader.readPage(path, keepFolder=os.path.join(tempfile.gettempdir(), "durhackHomr"))
                        newReads, trace["homrXml"], done = [(notes, ticksPerBeat)], xml, True
                    except Exception as error:
                        print("homr failed, using our own reader:", error, flush=True)
                if not done:
                    #the page is read at three working sizes and the reads voted on (reader/vote.py)
                    newReads = readVoted(path, trace)
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
        saveScan(path, notes, ticksPerBeat, decoded, trace, [str(warning.message) for warning in caught], readError)
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
