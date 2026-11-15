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
from .checksum import CHECK_BITS, Repair, decode_checked, encode_checked
from .identify import ScoreReading, decode_musicxml, decode_musicxml_with_style, decode_score, identify_chorale, matching_chorales
from .keys import Key
from .styles import STYLES, encode_style, style_capacity, style_weights
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
    "Repair",
    "ScoreReading",
    "Tune",
    "WeightState",
    "capacity_report",
    "decode",
    "decode_checked",
    "decode_musicxml",
    "decode_score",
    "decode_musicxml_with_style",
    "decode_phrase",
    "default_weights",
    "CHECK_BITS",
    "encode",
    "encode_checked",
    "encode_style",
    "encode_phrase",
    "identify_chorale",
    "phrase_capacity",
    "STYLES",
    "style_capacity",
    "style_weights",
    "read_musicxml",
    "load_tune",
    "matching_chorales",
    "options",
    "rejected_chorales",
    "start_state",
    "usable_chorales",
    "write_musicxml",
]
