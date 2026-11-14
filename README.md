# durhack2026

## renderer

Turns a MIDI file, or a list of notes, into a score image (png, svg or MusicXML). Other programs use it with `import renderer`.

Setup (python 3.10+, pip packages only):

```
pip install -r requirements.txt
python renderer/tests/apiTest.py
```

```python
import renderer

renderer.midiToScore("piece.mid", "score.png")
renderer.notesToScore([(60, 0, 480), (64, 480, 960)], 480, "score.png")
```

Also `python -m renderer --input piece.mid --output score.png`. Notes are `(pitch, startTick, endTick)`, plus an optional voice number. The score is a grand staff with any number of voices (see below).

Voices (for chorales): each MIDI track or channel is a voice, and chords inside a track are split into voices too, so four separate tracks, a two-hand piano file and one track of chords all give the same SATB score. The top half of the voices go on the treble staff and the rest on the bass staff, with stems up and down per staff.

Key, time signature and tempo are read from the MIDI (the first of each is used, later changes are ignored) and can be overridden: `renderer.midiToScore(path, key="Bb", timeSignature=(3, 4), tempo=80)`. Keys are named like `"G"`, `"Bb"` or `"F#m"`. Black keys are spelled with sharps or flats to suit the key, with naturals where needed. There is no pickup (anacrusis) bar.
