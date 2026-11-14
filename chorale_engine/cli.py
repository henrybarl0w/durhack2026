"""Command line: encode, decode, capacity, list."""

from __future__ import annotations

import argparse
import sys

import music21

from .codec import capacity_report, decode, encode
from .errors import CapacityError, ChoraleRejected, DecodeError
from .musicxml_io import read_musicxml, write_musicxml
from .options import rejected_chorales, usable_chorales


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="chorale-engine", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("encode", help="write a MusicXML score that encodes N")
    p.add_argument("--chorale", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("decode", help="print the integer written in a MusicXML score")
    p.add_argument("--chorale", type=int, required=True)
    p.add_argument("--in", dest="infile", required=True)
    p = sub.add_parser("capacity", help="per-phrase bits and total")
    p.add_argument("--chorale", type=int, required=True)
    sub.add_parser("list", help="music21 version, usable IDs, rejected IDs with reasons")
    args = parser.parse_args(argv)

    try:
        if args.command == "encode":
            write_musicxml(args.chorale, encode(args.chorale, args.n), args.out)
        elif args.command == "decode":
            print(decode(args.chorale, read_musicxml(args.chorale, args.infile)))
        elif args.command == "capacity":
            report = capacity_report(args.chorale)
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
