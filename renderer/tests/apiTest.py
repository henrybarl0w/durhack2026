import subprocess
import sys
import tempfile
import xml.etree.ElementTree as elementTree
from pathlib import Path

rendererFolder = Path(__file__).resolve().parent.parent
repoFolder = rendererFolder.parent
sys.path.insert(0, str(repoFolder))

import renderer
from renderer import Note
from stubEncoding import textToMidi

ticks = 480
variedNotes = [
    Note(60, 0, ticks),
    Note(61, ticks, ticks * 3),
    Note(48, ticks * 4, ticks * 4 + ticks // 2),
    Note(72, ticks * 5, ticks * 8),
    Note(55, ticks * 9, ticks * 13),
]


def checkTextToPng():
    score = renderer.midiToScore(textToMidi("hello"))
    return score[:8] == b"\x89PNG\r\n\x1a\n"


def checkEmptyMidi():
    return renderer.midiToScore(renderer.writeMidi([], ticks), format="svg").startswith("<svg")


def checkMusicXml():
    musicXml = renderer.midiToMusicXml(renderer.writeMidi(variedNotes, ticks))
    root = elementTree.fromstring(musicXml)
    measures = root.findall(".//measure")
    return len(measures) == 4 and len(root.findall(".//tie")) > 0


def checkRoundTripNotes():
    notes, ticksPerBeat = renderer.readNotes(renderer.writeMidi(variedNotes, ticks))
    return notes == variedNotes and ticksPerBeat == ticks


def checkWritesFile():
    with tempfile.TemporaryDirectory() as folderName:
        path = Path(folderName) / "score.svg"
        renderer.midiToScore(textToMidi("hi"), path)
        return path.read_text(encoding="utf-8").startswith("<svg")


def raisesRendererError(call):
    try:
        call()
    except renderer.RendererError:
        return True
    return False


def checkBadMidi():
    return raisesRendererError(lambda: renderer.midiToScore(b"not a midi file"))


def checkMissingFile():
    return raisesRendererError(lambda: renderer.midiToScore("does-not-exist.mid"))


def checkCommandLine():
    with tempfile.TemporaryDirectory() as folderName:
        midiPath = Path(folderName) / "piece.mid"
        pngPath = Path(folderName) / "score.png"
        midiPath.write_bytes(textToMidi("command line"))
        done = subprocess.run(
            [sys.executable, "-m", "renderer", "--input", str(midiPath), "--output", str(pngPath)],
            cwd=repoFolder,
            capture_output=True,
        )
        bad = subprocess.run(
            [sys.executable, "-m", "renderer", "--input", "missing.mid", "--output", str(pngPath)],
            cwd=repoFolder,
            capture_output=True,
        )
        return done.returncode == 0 and pngPath.read_bytes()[:4] == b"\x89PNG" and bad.returncode == 1


def checkMidiFolder():
    files = sorted((rendererFolder / "tests" / "midi").glob("*.mid"))
    return len(files) > 0 and all(renderer.midiToScore(path)[:4] == b"\x89PNG" for path in files)


def choraleXml(name):
    return renderer.midiToMusicXml(rendererFolder / "tests" / "midi" / (name + ".mid"))


def checkChoraleVoices():
    names = ["choraleFourTracks", "choralePianoTwoTracks", "choraleOneTrackChords"]
    xmls = [choraleXml(name) for name in names]
    voiceSets = [{voice.text for voice in elementTree.fromstring(xml).iter("voice")} for xml in xmls]
    return all(voices == {"1", "2", "3", "4"} for voices in voiceSets) and len(set(xmls)) == 1


def checkVoicesFromTracks():
    notes, _ = renderer.readNotes(rendererFolder / "tests" / "midi" / "choraleFourTracks.mid")
    return {note.voice for note in notes} == {0, 1, 2, 3}


def checkSingleVoiceUnchanged():
    root = elementTree.fromstring(renderer.midiToMusicXml(renderer.writeMidi(variedNotes, ticks)))
    return {voice.text for voice in root.iter("voice")} == {"1", "2"}


def midiPath(name):
    return rendererFolder / "tests" / "midi" / (name + ".mid")


def checkMetadataRead():
    metadata = renderer.readPiece(midiPath("choraleGMajor34"))[2]
    return metadata == renderer.Metadata((3, 4), "G", 80)


def checkKeyAndMeter():
    root = elementTree.fromstring(renderer.midiToMusicXml(midiPath("choraleGMajor34")))
    return root.find(".//fifths").text == "1" and root.find(".//beats").text == "3" and root.find(".//beat-type").text == "4"


def checkFlatsAndNaturals():
    xml = renderer.midiToMusicXml(midiPath("choraleFMajor"))
    return "<fifths>-1</fifths>" in xml and "<alter>-1</alter>" in xml and "<accidental>natural</accidental>" in xml


def checkMinorKeySharp():
    xml = renderer.midiToMusicXml(midiPath("aMinor68"))
    return "<mode>minor</mode>" in xml and "<accidental>sharp</accidental>" in xml and "<beats>6</beats>" in xml


def checkOverrides():
    xml = renderer.notesToMusicXml([(70, 0, 480)], 480, timeSignature=(2, 4), key="Bb", tempo=90)
    return "<fifths>-2</fifths>" in xml and "<beats>2</beats>" in xml and 'tempo="90"' in xml and "<step>B</step>" in xml


def checkBadKeyAndMeter():
    notes = [(60, 0, 480)]
    return raisesRendererError(lambda: renderer.notesToScore(notes, key="H")) and raisesRendererError(
        lambda: renderer.notesToScore(notes, timeSignature=(3, 5))
    )


checks = {
    "metadata read from midi": checkMetadataRead,
    "key and time signature in score": checkKeyAndMeter,
    "flat key with naturals": checkFlatsAndNaturals,
    "minor key and 6/8": checkMinorKeySharp,
    "key, meter and tempo overrides": checkOverrides,
    "bad key or time signature raises": checkBadKeyAndMeter,
    "chorale voices from 4 tracks, 2 tracks, 1 track": checkChoraleVoices,
    "voice numbers read from tracks": checkVoicesFromTracks,
    "single voice keeps two staves": checkSingleVoiceUnchanged,
    "every file in tests/midi": checkMidiFolder,
    "text to png": checkTextToPng,
    "empty midi": checkEmptyMidi,
    "musicxml structure": checkMusicXml,
    "midi note round trip": checkRoundTripNotes,
    "writes output file": checkWritesFile,
    "bad midi raises": checkBadMidi,
    "missing file raises": checkMissingFile,
    "command line": checkCommandLine,
}


def main():
    allPassed = True
    for name, check in checks.items():
        try:
            passed = check()
        except Exception as error:
            print(f"{name}: fail ({error})")
            allPassed = False
            continue
        allPassed = allPassed and passed
        print(f"{name}: {'pass' if passed else 'fail'}")
    sys.exit(0 if allPassed else 1)


main()
