"""Command line: encode, decode, capacity, list."""

from __future__ import annotations

import argparse
import sys

import music21

from .codec import capacity_report, decode, encode
from .errors import CapacityError, ChoraleRejected, DecodeError
from .identify import decode_musicxml_with_style
from .styles import STYLES, style_weights
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
    p = sub.add_parser("decode", help="print the integer written in a MusicXML score")
    p.add_argument("--chorale", type=int, help="optional: skip recognising the chorale")
    p.add_argument("--in", dest="infile", required=True)
    p = sub.add_parser("capacity", help="per-phrase bits and total")
    p.add_argument("--chorale", type=int, required=True)
    p.add_argument("--style", choices=STYLES, default="original")
    sub.add_parser("list", help="music21 version, usable IDs, rejected IDs with reasons")
    args = parser.parse_args(argv)

    try:
        if args.command == "encode":
            weights = style_weights(args.style)
            write_musicxml(args.chorale, encode(args.chorale, args.n, weights), args.out, style=args.style)
        elif args.command == "decode":
            if args.chorale is None:
                cid, n, style = decode_musicxml_with_style(args.infile)
                print(f"chorale {cid}, {style} style", file=sys.stderr)
                print(n)
            else:
                print(decode(args.chorale, read_musicxml(args.chorale, args.infile)))
        elif args.command == "capacity":
            report = capacity_report(args.chorale, style_weights(args.style))
            for row in report["phrases"]:
                print(f"phrase {row['phrase_index']}: {row['bits']} bits "
                      f"(capacity {row['capacity']}, min counts {row['min_counts']})")
            print(f"total: {report['bits']} bits (capacity {report['capacity']})")
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
