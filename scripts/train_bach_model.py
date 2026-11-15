"""Train the Bach weights model: integer log-frequency tables from Bach's own chords.

    python scripts/extract_bach_chords.py     # once, writes .cache/bach_chords.json
    python scripts/train_bach_model.py        # writes chorale_engine/data/bach_model.json
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chorale_engine.bach_weights import MODEL_PATH, chord_class, move_features, static_features  # noqa: E402
from chorale_engine.keys import Key  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / ".cache" / "bach_chords.json"
SCALE = 100
MIX = 0.7  # share of the chord-after-chord table against the chord-alone table


def log_int(p: float) -> int:
    return int(round(SCALE * math.log(p)))


def table(counts: collections.Counter) -> dict:
    """Laplace-smoothed log frequencies, with one value for anything unseen."""
    total = sum(counts.values())
    size = len(counts) + 1
    return {
        "seen": {k: log_int((n + 1) / (total + size)) for k, n in sorted(counts.items())},
        "unseen": log_int(1 / (total + size)),
    }


def main() -> int:
    data = json.loads(DATA.read_text())
    features = collections.defaultdict(collections.Counter)
    alone = collections.Counter()
    after = collections.defaultdict(collections.Counter)
    pairs = 0
    for cid in sorted(data, key=int):
        entry = data[cid]
        prev = None
        for (tonic, mode), chord in zip(entry["key"], entry["chords"]):
            if chord is None:
                prev = None
                continue
            tonic_pc = Key(tonic, mode).tonic_pc
            cls = chord_class(chord, tonic_pc)
            alone[cls] += 1
            for name, key in static_features(chord, tonic_pc):
                features[name][key] += 1
            if prev is not None:
                after[chord_class(prev, tonic_pc)][cls] += 1
                for name, key in move_features(prev, chord):
                    features[name][key] += 1
                pairs += 1
            prev = chord

    alone_total = sum(alone.values())
    alone_size = len(alone) + 1
    p_alone = {k: (n + 1) / (alone_total + alone_size) for k, n in alone.items()}
    model = {name: table(counts) for name, counts in sorted(features.items())}
    model["chord"] = {
        "alone": table(alone),
        "backoff": log_int(1 - MIX),
        "after": {
            prev: {
                cls: log_int(MIX * n / sum(row.values()) + (1 - MIX) * p_alone[cls])
                for cls, n in sorted(row.items())
            }
            for prev, row in sorted(after.items())
        },
    }
    model["meta"] = {"chorales": len(data), "chords": alone_total, "pairs": pairs, "scale": SCALE}
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(model, sort_keys=True, indent=0) + "\n")
    print(f"trained on {len(data)} chorales, {alone_total} chords, {pairs} chord pairs")
    print(f"{len(alone)} chord classes; wrote {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1024} KB)")
    moves = features["move1"]
    total = sum(moves.values())
    print("Bach's alto: holds its note on {:.0%} of beats".format(moves["0"] / total))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
