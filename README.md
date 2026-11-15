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

## chorale_engine

Hides an integer in a Bach-style SATB harmonization of a real chorale melody, and reads it back. The encoder takes an integer and a chorale ID (Riemenschneider number, music21 Bach corpus) and writes a four-part score as MusicXML. The decoder takes the same chorale ID and the chords actually written and returns the integer.

Setup (python 3.11+):

```
pip install -r requirements.txt
python -m pytest -q            # full suite
python -m pytest -q -m "not slow"
```

music21 is pinned to 10.5.2 in `pyproject.toml`. The home key comes from music21's key analysis, so the encoder and decoder machines must run the same version. Only chorales in 4/4 with four parts are accepted.

A chord is a tuple `(S, A, T, B)` of MIDI ints, one per crotchet beat where the soprano sounds.

### Public API

```python
import chorale_engine as ce

ce.usable_chorales()            # [6, 8, 10, ...] IDs the engine can encode into
ce.rejected_chorales()          # {1: "meter not 4/4 (3/4)", ...}

tune = ce.load_tune(6)          # Tune: beats, phrases, key map
tune.beats[0]                   # Beat(index, offset, soprano_midi, phrase_index, beat_index, is_cadence, length, key)
tune.phrases                    # ((0, 8), (8, 14), ...) beat ranges per phrase
tune.key_map                    # one Key per beat

prev_chord, prev_prev_bass = ce.start_state(6, 1)   # state at the first beat of phrase 1
ce.options(6, 1, 0, prev_chord, prev_prev_bass)     # the ordered option list at phrase 1, beat 0

ce.phrase_capacity(6, 1)        # 32: n fits phrase 1 when 0 <= n < 32
ce.capacity_report(6)           # {"phrases": [{"capacity", "bits", "min_counts"}, ...], "capacity", "bits"}

chords = ce.encode_phrase(6, 1, 17)   # chords of phrase 1 that encode 17
ce.decode_phrase(6, 1, chords)        # 17

chords = ce.encode(6, 123456)         # whole piece
ce.decode(6, chords)                  # 123456

ce.write_musicxml(6, chords, "score.musicxml")
ce.read_musicxml(6, "score.musicxml") # the same list of (S, A, T, B) tuples

ce.decode_musicxml("score.musicxml")  # (6, 123456): finds the chorale from the soprano, then decodes
ce.identify_chorale("score.musicxml") # 6
```

`start_state`, `options`, `phrase_capacity`, `capacity_report`, `encode_phrase`, `decode_phrase`, `encode` and `decode` all take an optional `weights=` function, `weights(state, options) -> list[int]`. The default is `ce.default_weights`. Encoder and decoder must use the same one.

To walk a phrase yourself: start from `start_state`, call `options` at each beat, pick one, and move to the state `(chosen, bass of the previous chord or None)`.

Errors: `CapacityError` (n negative or too large, the message gives the capacity), `DecodeError` (wrong number of chords, or a value at or above the capacity), `ChordNotInOptions` (a `DecodeError` with `phrase_index`, `beat_index`, `chord`, `options`), and `ChoraleRejected` (the chorale is not usable).

### Whole-piece encode and decode are for convenience and tests only

`encode` and `decode` split one integer across the phrases by mixed radix, so one misread phrase changes the entire integer. Error-tolerant pipelines (Reed-Solomon) must give each phrase its own bytes through `encode_phrase` and `decode_phrase`. Then a bad phrase costs only that phrase's symbols.

### MusicXML layout

The MusicXML is one part on a grand staff, so a whole chorale fits on one page. S (voice 1, stems up) and A (voice 2, stems down) are on the treble staff; T (voice 3, stems up) and B (voice 4, stems down) are on the bass staff. Fermatas are above the treble staff and below the bass staff. Every note off the key signature has its own printed accidental, even when the same note already had one earlier in the bar, so a reader never has to carry accidentals through a bar. There are no key-signature or clef changes. The soprano keeps its original rhythm. A, T and B get one note per beat, and the cadence chord is held for the fermata's length. Output is byte-identical for the same input. Render it with `renderer.writeScore(xml, "score.png", "png", 2000, 1.0)`.

Reading does not rely on voice numbers or stems. At each beat it takes the distinct pitches sounding on each staff: the higher is S (or T) and the lower is A (or B), because the voices never cross. A single notehead on a staff counts for both of its voices, so a note shared by S and A (or T and B) reads correctly whether it is written once or twice. A photo reader only needs to produce MusicXML with the right pitches on the right staff at each beat.

### Recognising the chorale

`decode_musicxml` and `chorale-engine decode` do not need the chorale ID. The soprano is Bach's tune, so the decoder compares the top treble note at every beat with each usable chorale and keeps the exact matches. Some tunes appear more than once in the corpus with the same soprano; then each match is tried as a decode. Where twins (for example 98 and 367) have the same keys, opening and cadences they encode every number identically, so the number is right and the lowest ID is reported. Different numbers from different chorales raise `DecodeError`.

### Styles (experimental)

The same number can be written two ways. `original` is the encoding described above. `fancy` uses weights trained on Bach's own chorales with stricter textbook rules (smoother lines, Bach-like harmony, a perfect cadence at the end where the tune allows), adds a walking bass and passing notes between the beats, and puts trills, turns and mordents on the soprano. The ornaments never change the notes on the beats.

```python
chords = ce.encode_style(16, n, "fancy")              # or "original"
ce.write_musicxml(16, chords, "score.musicxml", style="fancy")
ce.decode_musicxml_with_style("score.musicxml")       # (16, n, "fancy")
ce.decode_musicxml("score.musicxml")                  # (16, n): the style is worked out
```

`python -m chorale_engine encode --chorale 16 --n 123 --style fancy --out score.musicxml`; `decode` names the style it found. The fancy style holds fewer bits (66 of the 228 usable chorales hold three 30,000-word words, against 156 for the original), and 9 chorales cannot be written in it at all. The photo reader has not yet been tested on fancy scores.

### Checksum and repair

`checksum=True` (CLI `--checksum`) stores a 16-bit CRC with the number, so the decoder can repair up to two misread or unreadable chords. A misread chord is usually not one of its beat's options, which points at it; the decoder tries the other options there (and at every beat if nothing points anywhere) and keeps the result whose checksum matches. A reading that cannot be repaired is refused rather than guessed.

```python
chords = ce.encode_style(215, n, "fancy", checksum=True)
reading = ce.decode_score("score.musicxml", checksum=True)   # .n, .chorale_id, .style, .repaired
ce.decode_checked(215, chords_with_some_None)                # straight from chords (None = unreadable beat)
```

It costs 16 bits of capacity (`style_capacity(..., checksum=True)`). The web app uses it unless `CHORALE_CHECKSUM=0`, and packs the words as base-30001 digits (about 15 bits a word) to make room; scores made before still decode.

### Command line

```
python -m chorale_engine encode --chorale 6 --n 123456 --out score.musicxml
python -m chorale_engine decode --in score.musicxml              # finds the chorale itself
python -m chorale_engine decode --chorale 6 --in score.musicxml  # or give it
python -m chorale_engine capacity --chorale 6
python -m chorale_engine list
```

`chorale-engine` is the same command after `pip install -e .`. `scripts/survey.py` reports every chorale: usable or the rejection reason, phrases, bits, and rule blame for chorales with no legal path. `chorale_engine/checker.py` checks a written MusicXML file against the rules independently of the engine.

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

## webapp

A Flask app joining the other parts: type up to three words from the word list and get a chorale that hides them, upload the MusicXML to read them back, or point a camera at a chorale (on a screen or printed) to read the message from the picture.

```
pip install -r requirements.txt
python webapp/app.py
```

Then open http://127.0.0.1:5000. By default only the computer running it can reach the server, and Flask's debugger is on. A phone camera only works over https: run `python webapp/app.py --https`, open `https://<the computer's address>:5000` and accept the certificate warning. `--https` (and `--lan`, for http) opens the server to the network and turns the debugger off, since the debugger lets anyone who reaches it run code; only do it on a network you trust. Drawing a chorale takes a lock, because verovio is not thread safe.

Encode tab: the words become indices into `durhack_word_list.csv`, which `chorale_engine` hides in chorale 215 (with error correction when it fits, see the top of `webapp/app.py`). The result is a MusicXML file (the decodable one), a PNG of the score and a WAV.

Decode tab: upload the MusicXML and the words come back, with a note saying which damaged phrases the error correction repaired.

Scan tab (`webapp/scanner.py`): the browser sends a camera frame every 250 ms to `/scan/frame`. The server looks for the page's eight finder markers (four corners and one halfway along each edge) with `reader.rectify.findMarkersInGray`; the moment all eight are in view that frame is the snapshot (if only some are, after eight frames it takes the frame anyway, as long as the three corners it needs are there). The page then shows "reading the score" with a green frame and a running count of seconds while the server works on a background thread: the snapshot is straightened upright by its markers (`reader.rectify.warpByMarkers`), cleaned (`reader.rectify.cleanPage`) and read once by `reader.readScore`. A page takes about 15 seconds.

The notes then go through `decodeReads` in `webapp/chorales.py`: the chorale is identified from the page's soprano line, each phrase is rebuilt from the legal run of chords that fits the notes best (so a misread chord is mended to a legal one instead of upsetting the rest of the phrase), and the chords go to the same decoder as the Decode tab (`scan_words` in `app.py`). The page shows the words, the problems found and a MIDI download (`/scan/midi`). Note that this identification and repair look the tune up in the stored corpus; nothing is given to it, but it is a lookup.

Debug suite (`webapp/debugdump.py`): after every scan the server writes `webapp/debug/scan-<time>/` and a copy in `webapp/debug/latest/`: the snapshot, the read-back MIDI drawn again by the renderer, the two side by side, and `dump.txt` (step health, the bars that would not add up, and note by note which beats differ from the decoded chords and why) with the same facts as `dump.json`. `/debug` shows the newest one in the browser. It is not committed.

The first run builds `webapp/chorale_capacity.json` (capacity and beats of every usable chorale, about 30 seconds); the server then loads every chorale's tune in the background so the first scan is not slow. It is generated and not committed.

Edge cases and known limits:

- One scanner is shared by every browser, so two people scanning at once will mix their reads.
- Decoding is only as good as the reading. The repair fixes a chord only to a legal one, so a misread that is itself a legal chord goes through unchanged; error correction (chorale 215) is what recovers from a damaged phrase, and it is only available when chorale 215 has room for it.
- A photo of a screen or print needs the corner markers in view (the three it cannot do without; all eight give the flattest page) and a reasonably sharp, glare-free picture; a blurry frame loses the markers or the staff lines. The reader is cut for the renderer's own pictures, and a photo is much harder.
- Not yet shown to decode: a page of the new word-list scheme (pickup bar, dotted notes, ties, fermatas), photographed or rendered clean. Treat scanning the new pages as in progress.
