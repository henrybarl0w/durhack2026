"""Chorale engine: hide an integer in a Bach-style SATB harmonization.

Public API only. See README.md.
"""

from .codec import (
    capacity_report,
    decode,
    decode_phrase,
    encode,
    encode_phrase,
    phrase_capacity,
)
from .corpus import Beat, Tune, load_tune
from .errors import CapacityError, ChoraleRejected, ChordNotInOptions, DecodeError
from .keys import Key
from .musicxml_io import read_musicxml, write_musicxml
from .options import options, rejected_chorales, start_state, usable_chorales
from .weights import WeightState, default_weights

__all__ = [
    "Beat",
    "CapacityError",
    "ChoraleRejected",
    "ChordNotInOptions",
    "DecodeError",
    "Key",
    "Tune",
    "WeightState",
    "capacity_report",
    "decode",
    "decode_phrase",
    "default_weights",
    "encode",
    "encode_phrase",
    "phrase_capacity",
    "read_musicxml",
    "load_tune",
    "options",
    "rejected_chorales",
    "start_state",
    "usable_chorales",
    "write_musicxml",
]
