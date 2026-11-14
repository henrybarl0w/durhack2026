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
```

`start_state`, `options`, `phrase_capacity`, `capacity_report`, `encode_phrase`, `decode_phrase`, `encode` and `decode` all take an optional `weights=` function, `weights(state, options) -> list[int]`. The default is `ce.default_weights`. Encoder and decoder must use the same one.

To walk a phrase yourself: start from `start_state`, call `options` at each beat, pick one, and move to the state `(chosen, bass of the previous chord or None)`.

Errors: `CapacityError` (n negative or too large, the message gives the capacity), `DecodeError` (wrong number of chords, or a value at or above the capacity), `ChordNotInOptions` (a `DecodeError` with `phrase_index`, `beat_index`, `chord`, `options`), and `ChoraleRejected` (the chorale is not usable).

### Whole-piece encode and decode are for convenience and tests only

`encode` and `decode` split one integer across the phrases by mixed radix, so one misread phrase changes the entire integer. Error-tolerant pipelines (Reed-Solomon) must give each phrase its own bytes through `encode_phrase` and `decode_phrase`. Then a bad phrase costs only that phrase's symbols.

### MusicXML layout

The MusicXML has four separate staves, one voice per staff: parts `S`, `A`, `T`, `B` with clefs treble, treble, treble-8vb and bass. The PDF renderer must keep that layout for photo reading. The soprano keeps its original rhythm. A, T and B get one note per beat, and the cadence chord is held for the fermata's length. Output is byte-identical for the same input.

### Command line

```
python -m chorale_engine encode --chorale 6 --n 123456 --out score.musicxml
python -m chorale_engine decode --chorale 6 --in score.musicxml
python -m chorale_engine capacity --chorale 6
python -m chorale_engine list
```

`chorale-engine` is the same command after `pip install -e .`. `scripts/survey.py` reports every chorale: usable or the rejection reason, phrases, bits, and rule blame for chorales with no legal path. `chorale_engine/checker.py` checks a written MusicXML file against the rules independently of the engine.