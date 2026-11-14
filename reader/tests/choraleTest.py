#renders random chorales (the kind the encoder will make) at a few sizes, reads them back and compares
#run from the repo folder:
#   python reader/tests/choraleTest.py
#   python reader/tests/choraleTest.py --count 50 --first 100
import argparse
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

repoFolder = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repoFolder))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import reader
import renderer
from randomScore import chorale, keyTonics, ticksPerBeat

#(width, zoom): wide and narrow pages, small and large pictures
layouts = [(2000, 1.0), (1200, 1.0), (2000, 0.75), (2000, 2.0)]


def main():
    parser = argparse.ArgumentParser(description="round trip random chorales through the renderer and reader")
    parser.add_argument("--count", type=int, default=10, help="how many chorales")
    parser.add_argument("--first", type=int, default=1, help="first seed")
    parser.add_argument("--bars", type=int, default=6)
    arguments = parser.parse_args()

    exact = total = 0
    with tempfile.TemporaryDirectory() as folder:
        for seed in range(arguments.first, arguments.first + arguments.count):
            generator = random.Random(seed)
            key = generator.choice(sorted(keyTonics))
            notes = chorale(generator, arguments.bars, keyTonics[key])
            expected = Counter((pitch, start, end) for pitch, start, end, _ in notes)
            for width, zoom in layouts:
                total += 1
                imagePath = Path(folder) / f"chorale-{seed}-{width}-{zoom}.png"
                renderer.notesToScore(notes, ticksPerBeat, imagePath, width=width, zoom=zoom, key=key)
                try:
                    got = Counter(note[:3] for note in reader.readScore(imagePath)[0])
                except reader.ReaderError as error:
                    print(f"seed {seed} {key} width={width} zoom={zoom}: FAIL - {error}")
                    continue
                if got == expected:
                    exact += 1
                else:
                    missing = sorted((expected - got).elements())
                    extra = sorted((got - expected).elements())
                    print(f"seed {seed} {key} width={width} zoom={zoom}: FAIL - missed {missing[:4]}, wrong {extra[:4]}")
    print(f"\n{exact}/{total} chorale renders read back exactly")
    sys.exit(0 if exact == total else 1)


main()
