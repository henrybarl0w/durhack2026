## reader

Turns a score image made by the renderer back into notes, using OpenCV template matching. Only needs `opencv-python` and `numpy`.

```
python reader/tests/choraleTest.py
python reader/tests/endToEnd.py renderer/tests/midi/choraleFourTracks.mid --show
```

```python
import reader

notes, ticksPerBeat = reader.readScore("score.png")
midiBytes = reader.notesToMidi(notes, ticksPerBeat)
```

Also `python -m reader --input score.png --output piece.mid`. Notes come back as `(pitch, startTick, endTick, confidence)` at 480 ticks per beat; confidence (0 to 1) is how well the symbol matched and how squarely it sits on a line or space.

It reads 4/4 grand staff scores at any zoom or width: whole, half, quarter, eighth and sixteenth notes and rests, dotted notes and rests, chords up to 4 voices, sharps, flats, naturals and the key signature. Every size is measured in staff spaces, nothing is fixed in pixels. A bar that does not add up to 4 beats raises `reader.ReaderError`, so a misread is reported instead of passed on silently.

`choraleTest.py` renders random chorales (like the encoder makes) at several sizes, reads them back and compares. `endToEnd.py` does one file of your choice: renderer, reader, renderer again, then lists the notes that differ and writes `reader/tests/output/sideBySide.png` (original on top, read back below); `--show` pops it up, `--width` and `--zoom` change the render.

Templates are cut from the renderer's own output. If the renderer's look changes, rebuild them with `python -m reader.makeTemplates` (this one needs the renderer's requirements too).

Edge cases and known limits:

- Several voices of eighths and sixteenths with different rhythms in the same bar often fail (about half of random 2 to 4 voice pieces), with a "bar adds up to N beats" error rather than wrong notes. One voice of short notes, and chorales of quarters and halves, read correctly.
- A low bass note below the staff is occasionally missed (3 of 10 random chorales lost one bass note).
- Very small or very cramped scores (zoom 0.5, or width 500 with eighths) can fail the same way: the renderer squashes symbols into each other there.
- Only the first page: the renderer draws page 1 only, so a long piece is cut off before the reader sees it.
- The tempo mark is ignored; no tempo is written to the MIDI.
- Ticks per beat is always 480 and the output MIDI has a single track (voices are not kept apart).

### Improvements, switches and debugging

The reader has five improvements, each behind a switch in `reader/features.py` (`with features.using(curves=False): readScore(...)`):

| switch | what it does |
|---|---|
| `curves` | follows each staff across the page as a curve, so a bowed page does not shift notes by a line or space; kept per staff only if the heads sit closer to the line/space grid with it |
| `keyVote` | reads the key signature in every system and takes the commonest answer |
| `columns` | works each bar out a second way, with notes that sound together found across both staves, and keeps it only if fewer beats are missing |
| `restGuard` | drops a "rest" lying on top of strong noteheads |
| `vote` | reads the page at three sizes and votes note by note (`reader/vote.py`). Off by default: on the one page measured it scored worse than a single read and takes three times as long |

Two more switches cover what the new encoder writes: `pickup` (a short first bar is a pickup, and later bars start after its length) and `dots` (a dot after a notehead makes it half as long again).

The debugging can be stripped out. `reader/debug.py` records what each step measured, checks an invariant for it and catches a crash (falling back to the old code, recorded against that step); `reader/probe.py` is the reader's only link to it and has no-op fallbacks. To strip it, delete `reader/debug.py` and `webapp/debug*.py` (and `webapp/templates/debug.html` plus the `/debug` routes in `app.py` if you want the page gone too); the reader runs unchanged.

`python webapp/debugablate.py "<message>"` runs the reader on the newest saved scan with every step on and off and scores each run against what the page should say (it re-encodes the message to get that); `debugcompare.py` lists every missed, misread and invented note with the reason from the reader's trace. Pass the message exactly as it was typed.