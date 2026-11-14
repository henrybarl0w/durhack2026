import sys
from pathlib import Path

testsFolder = Path(__file__).resolve().parent
sys.path.insert(0, str(testsFolder.parent.parent))

import renderer

outputFolder = testsFolder / "output"
outputFolder.mkdir(exist_ok=True)
midiPath = testsFolder / "midi" / "twinkle.mid"

pngBytes = renderer.midiToScore(midiPath, outputFolder / "twinkle.png")
print("png written, starts with png signature:", pngBytes[:4] == b"\x89PNG")

svgText = renderer.midiToScore(midiPath, outputFolder / "twinkle.svg")
print("svg written, looks like svg:", svgText.startswith("<svg"))

musicXml = renderer.midiToScore(midiPath, outputFolder / "twinkle.musicxml")
print("musicxml written, looks like musicxml:", musicXml.startswith("<?xml"))

notes, ticksPerBeat = renderer.readNotes(midiPath)
print("notes read from midi:", len(notes))

notesScore = renderer.notesToScore([(60, 0, 480), (64, 480, 960), (67, 960, 1440)], 480, outputFolder / "notes.png")
print("notes to score works:", notesScore[:4] == b"\x89PNG")

try:
    renderer.midiToScore(testsFolder / "midi" / "missing.mid")
except renderer.RendererError as error:
    print("missing file raises RendererError:", error)

print("output files are in", outputFolder)
