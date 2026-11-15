# chorale_engine

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