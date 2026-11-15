# Bach Door (durhack2026)

Hide a short message inside a Bach chorale. The app writes the message into the harmony of a real chorale from the Riemenschneider collection, and gives back a printable score, a recording and a MusicXML file. The message can be read back from the MusicXML file, or from a photo of the printed page.

## How it works

Each chorale keeps Bach's melody in the soprano. The alto, tenor and bass are chosen beat by beat from the chords that are legal at that point: no parallel fifths or octaves, no voices crossing, no awkward leaps, and a proper cadence at the end of every phrase. At each beat there are several legal chords, and which one is chosen carries the data, like the digits of a number. The decoder finds the chorale from its melody, works out which chord was chosen at each beat and rebuilds the number.

A message is up to three words from a list of 30,000 (`durhack_word_list.csv`), or a line of printable ASCII text, which is split over as many chorales as it needs. Each message goes into a chorale picked at random from those with room for it.

There are two writing styles:

- **Default**: one chord per beat under the melody.
- **Fancy**: closer to Bach, with a walking bass, passing notes, suspensions, ornaments (trills, turns, mordents) and stronger cadences. It holds a little less.

With error correction on (the default), a 16-bit checksum is stored with the message. The decoder uses it to repair up to two misread chords, and to reject a page that would otherwise decode to the wrong words.

## Quick start

You need Python 3.11 or later.

```
pip install -r requirements.txt
python webapp/app.py
```

Open http://127.0.0.1:5000. The first start takes about 30 seconds while it measures how much each chorale can hold.

On a Mac, use `python3` and `pip3`. If port 5000 is taken, turn off AirPlay Receiver (System Settings, General, AirDrop & Handoff), which also uses it.

## Using the app

- **Compose**: type up to three words (or switch to text), choose Default or Fancy, and choose whether to use error correction. You get the score as a PNG to print, a WAV played on an organ, and the MusicXML file.
- **Decipher**: upload the MusicXML file (or the ZIP of a multiple MusicXML files) and the message comes back. The chorale, the style and the error correction are all recognised from the file.
- **Scan**: point a camera at the printed page, with all the squares around its edge in view. Reading a page takes about half a minute. Scan reads word messages only, not text.

## Using a phone

Run the app on a laptop and open it from the phone's browser. Both have to be on the same network.

```
python webapp/app.py --https
```

Open `https://<the laptop's address>:5000` on the phone and accept the certificate warning. The address is in the server's output, or run `ipconfig getifaddr en0` on a Mac. Phone browsers only allow the camera over https. `--lan` serves plain http, which is enough for Compose and Decipher.

Both flags open the server to the network and turn Flask's debugger off, since the debugger lets anyone who can reach it run code. Only use them on a network you trust. If the phone can't connect, the network may be blocking devices from reaching each other: use a phone hotspot instead.

## Settings

| Setting | What it does |
|---|---|
| `--lan` | Serve over http to the network |
| `--https` | Serve over https to the network, with a self-signed certificate |
| `CHORALE_CHECKSUM=0` | Make error correction off by default |
| `CHORALE_DEBUG=1` | Show the Scan tab's debugging to everyone |
| `?debug=1` on the address | Show the Scan tab's debugging in that browser |

## Project layout

| Folder | What it is |
|---|---|
| `chorale_engine/` | Encodes a number into a chorale's harmony and decodes it back; the voice-leading rules, both styles, the checksum and the MusicXML output |
| `renderer/` | Draws a score as a PNG or SVG with verovio |
| `reader/` | Reads the notes back from a picture of a score, with OpenCV |
| `webapp/` | The Flask app that joins them, and the Scan tab's decoder |
| `scripts/` | Tools for training the Bach model and measuring the music |
| `tests/` | The test suite |

Each folder has its own README with the details.

## Tests

```
python -m pytest -q
python -m pytest -q -m "not slow"
```

## Known limits

- music21 is pinned to 10.5.2. The decoder needs the same version as the encoder, because the key of each beat comes from music21's analysis.
- Only chorales in 4/4 with four parts are used.
- Scanning depends on the photo. With a perfect reading of the page every message decodes, so the limit is how well the reader handles real photos: glare, blur and curled paper.
- One scanner is shared by every browser, so two people scanning at once will mix their reads.