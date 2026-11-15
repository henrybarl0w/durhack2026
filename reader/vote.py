#step 5: the page is read at three working sizes and the reads are voted on, note by note. each size cuts the glyph
#templates, rounds the lines and finds the heads a little differently, so one size's misread is often right at another
import statistics
from concurrent.futures import ThreadPoolExecutor

from . import features
from .errors import ReaderError
from .probe import expect, guarded, probe
from .score import Note, readScore

#the shares of the measured blur that the three reads undo: the middle one is the normal share, the others a gentler and a firmer one. on the
#phone photo the measured blur was 4.9 pixels and widths of 2 and 3 pixels (shares 0.4 and 0.6) decoded where 2.5 and 3.5 did not
voteBlurs = (0.6, 0.4, 0.8)
#notes whose starts are this close (in ticks) are one chord
clusterTicks = 48


def readVoted(path, trace=None):
    #[(notes, ticksPerBeat), ...], the normal read first. with the vote switched on the page is read at three amounts of deblurring (an
    #amount that is right for one photo is too little or too much for another, and a chord wrong in one read is usually right in the
    #others); the decoder then takes a vote on every chord. the first read runs here (so its bar warnings are caught by the caller) and
    #the others on worker threads, quietly. a read that fails is skipped
    amounts = voteBlurs if features.on("vote") else (None,)
    with ThreadPoolExecutor(max_workers=max(1, len(amounts) - 1)) as pool:
        others = [pool.submit(readScore, path, None, None, True, amount) for amount in amounts[1:]]
        reads = [readScore(path, trace, None, False, amounts[0])]
        for index, future in enumerate(others):
            try:
                reads.append(future.result())
            except Exception as error:
                probe(trace, "vote", "skipped", blur=amounts[index + 1], error=f"{type(error).__name__}: {error}")
    return reads


def agreement(first, other):
    #the share of the first read's notes that the other read also has (same pitch, same start)
    found = {(note.pitch, round(note.start / clusterTicks)) for note in other}
    return sum((note.pitch, round(note.start / clusterTicks)) in found for note in first) / max(1, len(first))


def vote(reads):
    #notes with starts close together are one chord; each read has its own idea of what the chord holds. the chord
    #keeps as many notes as the middle read has, taking the pitches the reads back most strongly
    ticks = reads[0][1]
    tagged = sorted(((note.start, index, note) for index, (notes, _) in enumerate(reads) for note in notes), key=lambda t: t[0])
    clusters = []
    for item in tagged:
        if clusters and item[0] - clusters[-1][0][0] <= clusterTicks:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    result = []
    for cluster in clusters:
        perRead = [[note for _, index, note in cluster if index == r] for r in range(len(reads))]
        wanted = int(round(statistics.median(len(group) for group in perRead)))
        weight = {}
        for group in perRead:
            for pitch in {note.pitch for note in group}:
                weight[pitch] = weight.get(pitch, 0) + max(note.confidence for note in group if note.pitch == pitch) + 1
        for pitch in sorted(weight, key=lambda p: -weight[p])[:wanted]:
            members = [note for _, _, note in cluster if note.pitch == pitch]
            result.append(Note(pitch, int(statistics.median(n.start for n in members)),
                               int(statistics.median(n.end for n in members)),
                               round(sum(n.confidence for n in members) / len(members), 3)))
    result.sort(key=lambda note: (note.start, note.pitch))
    return result, ticks


def combine(reads, trace=None):
    #one (notes, ticksPerBeat) from all the reads; the first read alone if there is only one or the vote fails
    if len(reads) == 1:
        return reads[0]
    for index, other in enumerate(reads[1:], 1):
        share = agreement(reads[0][0], other[0])
        probe(trace, "vote", "agreement", withRead=index, share=round(share, 3), notes=[len(reads[0][0]), len(other[0])])
        expect(trace, "vote", share >= 0.5, "a read at another size shares under half its notes with the normal read",
               withRead=index, share=round(share, 3))
    return guarded(trace, "vote", lambda: vote(reads), lambda: reads[0])
