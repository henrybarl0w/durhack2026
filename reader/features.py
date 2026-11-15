#one switch per improvement to the reader, so any of them can be turned off to see what it did on its own
#(it can be used to ablate steps one at a time). all are on in normal use. in code:
#   with features.using(curves=False): readScore(...)
from contextlib import contextmanager

#1 curves: staff lines followed as curves across the page, not held level
#2 keyVote: the key signature read in every system and voted on, not taken from the first system alone
#3 columns: notes that sound together found across both staves of a system, tried beside the old per-staff way
#4 restGuard: a "rest" lying on top of strong noteheads is dropped before the bar is worked out
#5 vote: the page read at three sizes and the reads voted on note by note
#6 dots: a dot after a notehead makes it half as long again (the new encoder writes dotted notes)
#7 pickup: a short first bar is a pickup, and the bars after it start after its length (the new encoder's chorale starts mid-bar)
#8 nearLines: each notehead is measured against the staff lines actually beside it, not against the model line
#9 shortBars: a bar of fewer than four beats in the middle of a piece takes its real length (the new chorale has bars of 3 and 1 beats)
#10 deblur: the blur of a photographed page is measured from its staff lines and undone (Richardson-Lucy); a crisp page is left alone
#12 adaptiveBlur: how much blur to undo is found by trying it (the staff lines are measured after each trial and the search stops when they are as thin as a crisp render's), not guessed as a share of the measured width
#13 keyStrip: the key signature is looked for just after each clef, mark by mark at the heights it must have, and a notehead match there gives way to it
#11 weakSharps: a blurred sharp that scores a little under its threshold is accepted when a notehead sits just to its right
steps = ("curves", "keyVote", "columns", "restGuard", "vote", "dots", "pickup", "nearLines", "shortBars", "deblur", "weakSharps", "adaptiveBlur", "keyStrip")
active = {step: True for step in steps}
#the page is read at three amounts of deblurring and the reads vote on every chord. (reading at three sizes instead scored WORSE than one
#read on one scan, but with chords counted, not timed, reads at different blurs agree on the beats and disagree only on the notes)
active["vote"] = True
#the adaptive blur search never finds a width on real photos (the lines stay about 12 pixels wide through every trial) and costs about 9 seconds a read,
#so it is off until it works; the fixed-share deblur it falls back to gives the same read
active["adaptiveBlur"] = False


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
