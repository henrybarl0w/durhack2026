"""Command line: encode, decode, capacity, list."""

from __future__ import annotations

import argparse
import sys

import music21

from .codec import capacity_report, decode, encode
from .errors import CapacityError, ChoraleRejected, DecodeError
from .checksum import decode_checked
from .identify import decode_score
from .styles import STYLES, encode_style, style_capacity, style_weights
from .musicxml_io import read_musicxml, write_musicxml
from .options import rejected_chorales, usable_chorales


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="chorale-engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("encode", help="write a MusicXML score that encodes N")
    p.add_argument("--chorale", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--style", choices=STYLES, default="original", help="original encoding or the fancier writing")
    p.add_argument("--checksum", action="store_true", help="add a 16-bit checksum so misread chords can be repaired")
    p = sub.add_parser("decode", help="print the integer written in a MusicXML score")
    p.add_argument("--chorale", type=int, help="optional: skip recognising the chorale")
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--checksum", action="store_true", help="the score was written with --checksum: check and repair")
    p = sub.add_parser("capacity", help="per-phrase bits and total")
    p.add_argument("--chorale", type=int, required=True)
    p.add_argument("--style", choices=STYLES, default="original")
    p.add_argument("--checksum", action="store_true", help="also show what is left for the message after the checksum")
    sub.add_parser("list", help="music21 version, usable IDs, rejected IDs with reasons")
    args = parser.parse_args(argv)

    try:
        if args.command == "encode":
            chords = encode_style(args.chorale, args.n, args.style, checksum=args.checksum)
            write_musicxml(args.chorale, chords, args.out, style=args.style)
        elif args.command == "decode":
            if args.chorale is None:
                reading = decode_score(args.infile, checksum=args.checksum)
                note = f"chorale {reading.chorale_id}, {reading.style} style"
                if reading.repaired:
                    note += ", repaired " + ", ".join(f"phrase {p} beat {b}" for p, b in reading.repaired)
                print(note, file=sys.stderr)
                print(reading.n)
            elif args.checksum:
                print(decode_checked(args.chorale, read_musicxml(args.chorale, args.infile)).n)
            else:
                print(decode(args.chorale, read_musicxml(args.chorale, args.infile)))
        elif args.command == "capacity":
            report = capacity_report(args.chorale, style_weights(args.style))
            for row in report["phrases"]:
                print(f"phrase {row['phrase_index']}: {row['bits']} bits "
                      f"(capacity {row['capacity']}, min counts {row['min_counts']})")
            print(f"total: {report['bits']} bits (capacity {report['capacity']})")
            if args.checksum:
                checked = style_capacity(args.chorale, args.style, checksum=True)
                print(f"with checksum: {checked['bits']} bits for the message "
                      f"(capacity {checked['capacity']}, {checked['check_bits']} check bits)")
        else:
            print(f"music21 {music21.__version__}")
            usable = usable_chorales()
            print(f"usable ({len(usable)}): {' '.join(str(c) for c in usable)}")
            rejected = rejected_chorales()
            print(f"rejected ({len(rejected)}):")
            for cid, reason in rejected.items():
                print(f"  {cid}: {reason}")
    except (CapacityError, DecodeError, ChoraleRejected, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
