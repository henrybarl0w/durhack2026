"""realise: turn the chosen chords into the final notes.

For now this is the identity. It must stay a pure function of the choices.
MusicXML export goes through it.
"""

from __future__ import annotations


def realise(choices) -> list[tuple[int, int, int, int]]:
    return [tuple(int(m) for m in chord) for chord in choices]
