#one switch per improvement to the reader, so any of them can be turned off to see what it did on its own
#(webapp/debugablate.py does exactly that). all are on in normal use. in code:
#   with features.using(curves=False): readScore(...)
from contextlib import contextmanager

#1 curves: staff lines followed as curves across the page, not held level
#2 keyVote: the key signature read in every system and voted on, not taken from the first system alone
#3 columns: notes that sound together found across both staves of a system, tried beside the old per-staff way
#4 restGuard: a "rest" lying on top of strong noteheads is dropped before the bar is worked out
#5 vote: the page read at three sizes and the reads voted on note by note
steps = ("curves", "keyVote", "columns", "restGuard", "vote")
active = {step: True for step in steps}
#measured on one scan: reading at three sizes scored 4 exact notes WORSE than one read and takes three times as long,
#so it is off until a measurement says otherwise. webapp/debugablate.py shows its effect
active["vote"] = False


def on(step):
    return active.get(step, True)


@contextmanager
def using(**switches):
    saved = dict(active)
    active.update(switches)
    try:
        yield
    finally:
        active.clear()
        active.update(saved)
