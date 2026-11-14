import os

from .errors import RendererError
from .midiFile import Note, readNotes
from .notation import buildMusicXml
from .rendering import musicXmlToSvg, svgToPng

formatsByExtension = {".png": "png", ".svg": "svg", ".musicxml": "musicxml", ".xml": "musicxml"}
defaultTicksPerBeat = 480


def notesToMusicXml(notes, ticksPerBeat=defaultTicksPerBeat):
    return buildMusicXml([Note(*note) for note in notes], ticksPerBeat)


def midiToMusicXml(midiSource):
    notes, ticksPerBeat = readNotes(midiSource)
    return notesToMusicXml(notes, ticksPerBeat)


def notesToScore(notes, ticksPerBeat=defaultTicksPerBeat, outputPath=None, format=None, width=2000, zoom=1.0):
    if format is None:
        extension = os.path.splitext(str(outputPath))[1].lower() if outputPath else ""
        format = formatsByExtension.get(extension, "png")
    if format not in ("png", "svg", "musicxml"):
        raise RendererError("unknown format: " + str(format))

    musicXml = notesToMusicXml(notes, ticksPerBeat)
    if format == "musicxml":
        score = musicXml
    else:
        svg = musicXmlToSvg(musicXml, width)
        score = svg if format == "svg" else svgToPng(svg, zoom)

    if outputPath is not None:
        if isinstance(score, str):
            with open(outputPath, "w", encoding="utf-8") as outputFile:
                outputFile.write(score)
        else:
            with open(outputPath, "wb") as outputFile:
                outputFile.write(score)
    return score


def midiToScore(midiSource, outputPath=None, format=None, width=2000, zoom=1.0):
    notes, ticksPerBeat = readNotes(midiSource)
    return notesToScore(notes, ticksPerBeat, outputPath, format, width, zoom)
