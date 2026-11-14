import os

from .errors import RendererError
from .midiFile import Note, readPiece
from .notation import buildMusicXml
from .rendering import musicXmlToSvg, svgToPng

formatsByExtension = {".png": "png", ".svg": "svg", ".musicxml": "musicxml", ".xml": "musicxml"}
defaultTicksPerBeat = 480


def notesToMusicXml(notes, ticksPerBeat=defaultTicksPerBeat, timeSignature=None, key=None, tempo=None):
    return buildMusicXml(
        [Note(*note) for note in notes], ticksPerBeat, tuple(timeSignature or (4, 4)), key or "C", tempo
    )


def midiToMusicXml(midiSource, timeSignature=None, key=None, tempo=None):
    notes, ticksPerBeat, metadata = readPiece(midiSource)
    return notesToMusicXml(
        notes, ticksPerBeat, timeSignature or metadata.timeSignature, key or metadata.key, tempo or metadata.tempo
    )


def writeScore(musicXml, outputPath, format, width, zoom):
    if format is None:
        extension = os.path.splitext(str(outputPath))[1].lower() if outputPath else ""
        format = formatsByExtension.get(extension, "png")
    if format not in ("png", "svg", "musicxml"):
        raise RendererError("unknown format: " + str(format))

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


def notesToScore(
    notes, ticksPerBeat=defaultTicksPerBeat, outputPath=None, format=None, width=2000, zoom=1.0,
    timeSignature=None, key=None, tempo=None,
):
    musicXml = notesToMusicXml(notes, ticksPerBeat, timeSignature, key, tempo)
    return writeScore(musicXml, outputPath, format, width, zoom)


def midiToScore(
    midiSource, outputPath=None, format=None, width=2000, zoom=1.0, timeSignature=None, key=None, tempo=None,
):
    musicXml = midiToMusicXml(midiSource, timeSignature, key, tempo)
    return writeScore(musicXml, outputPath, format, width, zoom)
