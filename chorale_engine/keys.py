"""Keys: home key (full-score analysis), candidate keys and the per-beat key map.

Keys are stored by tonic name and mode. All arithmetic uses the line of
fifths (F=-1, C=0, G=1, ...), so spelling and pitch classes are plain ints.
"""

from __future__ import annotations

from dataclasses import dataclass

_LETTERS = "FCGDAEB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Line-of-fifths offsets of scale degrees 1..7 from the tonic.
_MAJOR_DEGREES = (0, 2, 4, -1, 1, 3, 5)
_MINOR_DEGREES = (0, 2, -3, -1, 1, -4, -2)  # natural minor
_RAISED_6 = 3
_RAISED_7 = 5


def name_to_lof(name: str) -> int:
    """'C' -> 0, 'F#' -> 6, 'B-' -> -2. Flats are '-' or 'b'."""
    letter = name[0].upper()
    alter = 0
    for ch in name[1:]:
        if ch == "#":
            alter += 1
        elif ch in "-b":
            alter -= 1
        else:
            raise ValueError(f"bad note name {name!r}")
    return _LETTERS.index(letter) - 1 + 7 * alter


def lof_to_step_alter(lof: int) -> tuple[str, int]:
    """Line-of-fifths index -> (letter, alter)."""
    return _LETTERS[(lof + 1) % 7], (lof + 1) // 7


def lof_to_name(lof: int) -> str:
    step, alter = lof_to_step_alter(lof)
    return step + ("#" * alter if alter > 0 else "-" * (-alter))


def lof_to_pc(lof: int) -> int:
    return (lof * 7) % 12


@dataclass(frozen=True)
class Key:
    """A key: tonic name ('C', 'F#', 'B-') and mode ('major' or 'minor')."""

    tonic: str
    mode: str

    def __post_init__(self):
        if self.mode not in ("major", "minor"):
            raise ValueError(f"unsupported mode {self.mode!r}")
        name_to_lof(self.tonic)  # validates the name

    @property
    def lof(self) -> int:
        return name_to_lof(self.tonic)

    @property
    def tonic_pc(self) -> int:
        return lof_to_pc(self.lof)

    @property
    def is_minor(self) -> bool:
        return self.mode == "minor"

    def degree_lofs(self) -> tuple[int, ...]:
        """Line-of-fifths indices of every note of the key, in a fixed order.

        Major: degrees 1..7. Minor: natural degrees 1..7, raised 6th, raised 7th.
        """
        t = self.lof
        if self.mode == "major":
            return tuple(t + d for d in _MAJOR_DEGREES)
        return tuple(t + d for d in _MINOR_DEGREES) + (t + _RAISED_6, t + _RAISED_7)

    def pitch_classes(self) -> frozenset[int]:
        return frozenset(lof_to_pc(i) for i in self.degree_lofs())

    def __str__(self) -> str:
        return f"{self.tonic} {self.mode}"


def candidate_keys(home: Key) -> tuple[Key, ...]:
    """Home, dominant, subdominant, relative, relative's dominant, relative's subdominant.

    Dominant and subdominant keep the mode of the key they belong to.
    The relative has the other mode.
    """
    t = home.lof
    if home.mode == "major":
        other, rel = "minor", t + 3
    else:
        other, rel = "major", t - 3
    return (
        home,
        Key(lof_to_name(t + 1), home.mode),
        Key(lof_to_name(t - 1), home.mode),
        Key(lof_to_name(rel), other),
        Key(lof_to_name(rel + 1), other),
        Key(lof_to_name(rel - 1), other),
    )


def home_tonic_pcs(home: Key) -> frozenset[int]:
    """Pitch classes of the home tonic triad. In minor the major 3rd (Picardy) counts too."""
    t = home.tonic_pc
    if home.mode == "major":
        return frozenset({t, (t + 4) % 12, (t + 7) % 12})
    return frozenset({t, (t + 3) % 12, (t + 4) % 12, (t + 7) % 12})


def home_key_from_score(score) -> Key:
    """Home key = music21 analyze('key') on the full score, all parts."""
    k = score.analyze("key")
    return Key(k.tonic.name, k.mode)


class NoKeyMap(ValueError):
    """No key assignment covers every soprano pitch class."""


def key_map(soprano_pcs, home: Key) -> list[Key]:
    """Assign one candidate key to every beat.

    Every soprano pitch class must be in its beat's key. The assignment
    minimises (number of key changes, number of beats not in the home key)
    lexicographically. Ties are broken left to right: among keys that still
    allow an optimal completion, keep the previous beat's key if possible,
    otherwise take the first key in candidate order.
    """
    pcs = [int(p) % 12 for p in soprano_pcs]
    cands = candidate_keys(home)
    nk = len(cands)
    sets = [k.pitch_classes() for k in cands]
    n = len(pcs)
    if n == 0:
        return []
    for t, pc in enumerate(pcs):
        if not any(pc in s for s in sets):
            raise NoKeyMap(f"soprano pitch class {pc} at beat {t} is in no candidate key")

    # cost[t][k] = best (changes, non-home beats) for beats t.. given key k at t,
    # or None if key k cannot hold beat t.
    cost: list[list[tuple[int, int] | None]] = [[None] * nk for _ in range(n)]
    for t in range(n - 1, -1, -1):
        for k in range(nk):
            if pcs[t] not in sets[k]:
                continue
            own = 0 if k == 0 else 1
            if t == n - 1:
                cost[t][k] = (0, own)
                continue
            best = None
            for k2 in range(nk):
                c2 = cost[t + 1][k2]
                if c2 is None:
                    continue
                cand = (c2[0] + (0 if k2 == k else 1), c2[1] + own)
                if best is None or cand < best:
                    best = cand
            cost[t][k] = best

    feasible = [c for c in cost[0] if c is not None]
    if not feasible:
        raise NoKeyMap("no key assignment covers every soprano pitch class")
    target = min(feasible)
    cur = next(k for k in range(nk) if cost[0][k] == target)
    chosen = [cur]
    for t in range(1, n):
        own = 0 if cur == 0 else 1
        # What the rest (beats t..) may cost, including the change into beat t.
        rest = (target[0], target[1] - own)

        def total(k2):
            c2 = cost[t][k2]
            if c2 is None:
                return None
            return (c2[0] + (0 if k2 == cur else 1), c2[1])

        if total(cur) == rest:
            nxt = cur
        else:
            nxt = next(k2 for k2 in range(nk) if total(k2) == rest)
        target = cost[t][nxt]
        cur = nxt
        chosen.append(cur)
    return [cands[k] for k in chosen]
