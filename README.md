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

Also `python -m renderer --input piece.mid --output score.png`. Notes are `(pitch, startTick, endTick)`. The score is one voice on a 4/4 grand staff, and tempo, key and time signature in the MIDI are ignored.
