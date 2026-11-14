#live camera scanning: find the corner markers in each frame, and once they hold steady,
#warp a few frames upright, merge them to wipe out glare, and run the reader on the result
import os
import sys
import tempfile
import time
import warnings

import cv2
import numpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reader import ReaderError, notesToMidi, readScore
from chorales import decodeNotes
from reader.rectify import cleanPage, findMarkersInGray, resizeLongSide, warpByMarkers

#frames are looked at about this size for the cheap marker check
checkSide = 1200
#markers in this many frames before collecting starts
steadyNeeded = 2
#frames in a row without markers before starting over
missesAllowed = 4
#frames merged into one picture
mergeCount = 4
#a gap longer than this (seconds) starts over
staleAfter = 3.0

pitchNames = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pitchName(pitch):
    return pitchNames[pitch % 12] + str(pitch // 12 - 1)


class Scanner:
    def __init__(self):
        self.reset()

    def reset(self):
        self.steady = 0
        self.misses = 0
        self.frames = []
        self.lastTime = 0
        self.result = None

    def addFrame(self, jpegBytes):
        now = time.time()
        if now - self.lastTime > staleAfter:
            self.steady, self.frames = 0, []
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
                self.steady, self.frames = 0, []
            if self.frames or self.steady:
                return {"status": f"lost the markers, keep holding still ({len(self.frames)}/{mergeCount})", "found": False}
            return {"status": "markers not found", "found": False}
        self.misses = 0
        self.steady += 1
        corners = [list(map(float, point / gray.shape[1])) for point in markers[:3]]
        if self.steady < steadyNeeded:
            return {"status": "markers found, keep holding still", "found": True, "corners": corners}
        #warp each frame upright from the full size picture (the small copy was only for finding the markers):
        #markers line the frames up, so merging is per pixel
        factor = max(image.shape) / max(gray.shape)
        bigMarkers = (markers[0] * factor, markers[1] * factor, markers[2] * factor, markers[3] * factor)
        self.frames.append(warpByMarkers(image, bigMarkers))
        if len(self.frames) < mergeCount:
            return {"status": f"keep holding still ({len(self.frames)}/{mergeCount})", "found": True, "corners": corners}
        frames, self.frames, self.steady = self.frames, [], 0
        result = self.readMerged(frames)
        result["corners"] = corners
        return result

    def readMerged(self, frames):
        #warps come out nearly the same size; trim to the smallest
        height = min(frame.shape[0] for frame in frames)
        width = min(frame.shape[1] for frame in frames)
        stack = numpy.stack([frame[:height, :width] for frame in frames])
        #glare brightens, ink darkens: the darkest value of each pixel is ink wherever any frame saw it.
        #the second darkest is a little safer against one frame's motion blur or a shadow
        merged = numpy.sort(stack, axis=0)[min(1, len(frames) - 1)] if len(frames) > 2 else stack.min(axis=0)
        steps = {}
        cleanPage(merged, True, steps)
        path = os.path.join(tempfile.gettempdir(), "durhackScan.png")
        cv2.imwrite(path, steps["result"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                notes, ticksPerBeat = readScore(path)
            except ReaderError as error:
                return {"status": "could not read: " + str(error), "found": True, "done": True, "error": str(error)}
        self.result = (notes, ticksPerBeat)
        #notes back to text: the chorale is identified, misread chords are snapped to legal ones, then decoded
        try:
            decoded = decodeNotes(notes, ticksPerBeat)
        except Exception as error:
            decoded = {"error": str(error) or type(error).__name__}
            print("decode failed:", decoded["error"], f"({len(notes)} notes read)", flush=True)
        return {
            "decoded": decoded,
            "status": "read",
            "found": True,
            "done": True,
            "notes": [
                {"name": pitchName(note.pitch), "pitch": note.pitch, "start": note.start, "end": note.end,
                 "confidence": note.confidence}
                for note in notes
            ],
            "ticksPerBeat": ticksPerBeat,
            "warnings": [str(warning.message) for warning in caught],
        }

    def midi(self):
        if self.result is None:
            return None
        notes, ticksPerBeat = self.result
        return notesToMidi(notes, ticksPerBeat)
