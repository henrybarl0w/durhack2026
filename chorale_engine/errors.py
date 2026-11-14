"""Exceptions raised by the chorale engine."""


class ChoraleRejected(ValueError):
    """The chorale cannot be used by the engine. The message is the reason."""

    def __init__(self, chorale_id, reason):
        super().__init__(f"chorale {chorale_id} rejected: {reason}")
        self.chorale_id = chorale_id
        self.reason = reason


class CapacityError(ValueError):
    """n is negative or too large for the phrase or piece."""


class DecodeError(ValueError):
    """The written chords cannot be decoded."""


class ChordNotInOptions(DecodeError):
    """A written chord is not in the option list of its beat."""

    def __init__(self, phrase_index, beat_index, chord, options):
        super().__init__(
            f"chord {tuple(chord)} at phrase {phrase_index}, beat {beat_index} "
            f"is not one of the {len(options)} options for that beat"
        )
        self.phrase_index = phrase_index
        self.beat_index = beat_index
        self.chord = chord
        self.options = options
