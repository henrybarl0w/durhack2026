# webapp

A Flask app joining the other parts: type up to three words from the word list and get a chorale that hides them, upload the MusicXML to read them back, or point a camera at a chorale (on a screen or printed) to read the message from the picture.

```
pip install -r requirements.txt
python webapp/app.py
```

Then open http://127.0.0.1:5000. By default only the computer running it can reach the server, and Flask's debugger is on. A phone camera only works over https: run `python webapp/app.py --https`, open `https://<the computer's address>:5000` and accept the certificate warning. `--https` (and `--lan`, for http) opens the server to the network and turns the debugger off, since the debugger lets anyone who reaches it run code; only do it on a network you trust. Drawing a chorale takes a lock, because verovio is not thread safe.

Compose tab: up to three words from `durhack_word_list.csv`, or printable ASCII text (split over as many chorales as it needs). Each message goes into a chorale picked at random from those with room for it, in the chosen style: Default, or Fancy (walking bass, passing notes, suspensions, ornaments). With error correction on (the default) a 16-bit checksum is stored with the words, so up to two misread chords can be repaired (`chorale_engine/checksum.py`). The result is a MusicXML file (the decodable one), a PNG of the score and a WAV played on an organ (`SOUND` in `app.py` also offers harpsichord and strings).

Decipher tab: upload the MusicXML (or the ZIP of an ASCII message) and the message comes back. The chorale, the style and whether the checksum is on are all recognised from the file, with a note saying which chords the checksum repaired.

Scan tab (`webapp/scanner.py`): the browser sends a camera frame every 250 ms to `/scan/frame`. The server looks for the page's eight finder markers (four corners and one halfway along each edge) with `reader.rectify.findMarkersInGray`; the moment all eight are in view that frame is the snapshot (if only some are, after eight frames it takes the frame anyway, as long as the three corners it needs are there). The page then shows "reading the score" with a green frame and a running count of seconds while the server works on a background thread: the snapshot is straightened upright by its markers (`reader.rectify.warpByMarkers`), cleaned (`reader.rectify.cleanPage`) and read once by `reader.readScore`. A page takes about 15 seconds.

The notes then go through `decodeReads` in `webapp/chorales.py`. The chorale is identified from the page's soprano line; where the collection sets the same tune more than once, each match is tried. Each phrase is rebuilt from the legal run of chords that fits the notes best, in each style (so a misread chord is mended to a legal one instead of upsetting the rest of the phrase; in Fancy a suspension's held note counts as the chord note it falls to). The chords go to the same decoder as the Decipher tab (`scan_words` in `app.py`), whose checksum mends up to two chords and chooses between repairs that fit the page equally well. With a perfect reading of the page every message decodes, in both styles. Scan reads word messages only, not ASCII text. The page shows the words, the problems found and a MIDI download (`/scan/midi`).

The Scan tab's debugging (the chorale and how many chords were corrected, the problems found, the notes read and a MIDI of the reading) is hidden. Add `?debug=1` to the page's address to see it, or start the server with `CHORALE_DEBUG=1` to show it to everyone.

Debug suite (`webapp/debugdump.py`): after every scan the server writes `webapp/debug/scan-<time>/` and a copy in `webapp/debug/latest/`: the snapshot, the read-back MIDI drawn again by the renderer, the two side by side, and `dump.txt` (step health, the bars that would not add up, and note by note which beats differ from the decoded chords and why) with the same facts as `dump.json`. `/debug` shows the newest one in the browser. It is not committed.

The first run builds `webapp/chorale_capacity.json` (capacity and beats of every usable chorale, about 30 seconds); the server then loads every chorale's tune in the background so the first scan is not slow. It is generated and not committed.

Edge cases and known limits:

- One scanner is shared by every browser, so two people scanning at once will mix their reads.
- A photo of a screen or print needs the corner markers in view (the three it cannot do without; all eight give the flattest page) and a reasonably sharp, glare-free picture; a blurry frame loses the markers or the staff lines. The reader is cut for the renderer's own pictures, and a photo is much harder.
- Not yet shown to decode: a page of the new word-list scheme (pickup bar, dotted notes, ties, fermatas), photographed or rendered clean. Treat scanning the new pages as in progress.