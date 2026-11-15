"""Two ways of writing the same number: the original encoding, and the fancier writing.

    "original"  the locked method: phase 1 weights, plain chords, one note per beat in
                alto, tenor and bass. Exactly what the engine has always written.
    "fancy"     the experimental writing: Bach-trained weights with the strict style
                rules (smoother lines, Bach-like harmony, perfect final cadence), a
                walking bass and passing notes between beats, and trills, turns and
                mordents on the soprano.

The two styles turn a number into different chords, so a score must be decoded with
the style it was written in. decode_musicxml works the style out by itself.

    chords = encode_style(16, n, "fancy")
    write_musicxml(16, chords, "score.musicxml", style="fancy")
    decode_musicxml("score.musicxml")              # (16, n), whichever style
    decode_musicxml_with_style("score.musicxml")   # (16, n, "fancy")

Both styles take checksum=True: 16 bits of the capacity then hold a checksum, and the
decoder can repair up to two misread chords (decode with checksum=True as well).
"""

from __future__ import annotations

from .codec import capacity_report, encode
from .weights import default_weights

STYLES = ("original", "fancy")


def style_weights(style: str):
    """The weights function a style encodes and decodes with."""
    if style == "original":
        return default_weights
    if style == "fancy":
        from .style import strict_weights

        return strict_weights
    raise ValueError(f"unknown style {style!r}: use one of {', '.join(STYLES)}")


def style_writing(style: str) -> dict:
    """The MusicXML writer options of a style: off-beat ornaments and ornament signs."""
    style_weights(style)  # validates the name
    fancy = style == "fancy"
    return {"ornamented": fancy, "signs": fancy}


def encode_style(chorale_id: int, n: int, style: str = "original", checksum: bool = False) -> list:
    """Chords for n in the style. checksum=True adds the 16-bit checksum that lets the decoder
    repair misread chords (see checksum.py); decode with checksum=True too."""
    if checksum:
        from .checksum import encode_checked

        return encode_checked(chorale_id, n, style_weights(style))
    return encode(chorale_id, n, style_weights(style))


def style_capacity(chorale_id: int, style: str = "original", checksum: bool = False) -> dict:
    """{"capacity", "bits"} for the style: n fits when n < capacity. With checksum=True this is
    what is left for the message once the 16 check bits are taken (plus the full report under
    "report")."""
    report = capacity_report(chorale_id, style_weights(style))
    if not checksum:
        return report
    from .checksum import CHECK_BITS

    cap = report["capacity"] >> CHECK_BITS
    return {"capacity": cap, "bits": max(cap.bit_length() - 1, 0), "check_bits": CHECK_BITS, "report": report}
