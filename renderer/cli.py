import argparse
import sys

from .api import midiToScore
from .errors import RendererError


def main():
    parser = argparse.ArgumentParser(prog="renderer", description="turn a midi file into a score")
    parser.add_argument("--input", required=True, help="path to the .mid file")
    parser.add_argument("--output", required=True, help="score path, .png .svg or .musicxml")
    parser.add_argument("--format", choices=["png", "svg", "musicxml"], help="default: from the output extension")
    parser.add_argument("--width", type=int, default=2000, help="score width in verovio units")
    parser.add_argument("--zoom", type=float, default=1.0, help="png scale factor")
    arguments = parser.parse_args()

    try:
        score = midiToScore(arguments.input, arguments.output, arguments.format, arguments.width, arguments.zoom)
    except (RendererError, OSError) as error:
        print("error: " + str(error), file=sys.stderr)
        sys.exit(1)
    print(f"wrote {arguments.output} ({len(score)} {'chars' if isinstance(score, str) else 'bytes'})")
