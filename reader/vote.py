#step 5: the page is read at three working sizes and the reads are voted on, note by note. each size cuts the glyph
#templates, rounds the lines and finds the heads a little differently, so one size's misread is often right at another
import statistics
import threading
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

from . import features
from .errors import ReaderError
from .probe import expect, guarded, probe
from .score import Note, readScore

#the shares of the measured blur that the three reads undo: the middle one is the normal share, the others a gentler and a firmer one. on the
#phone photo the measured blur was 4.9 pixels and widths of 2 and 3 pixels (shares 0.4 and 0.6) decoded where 2.5 and 3.5 did not
voteBlurs = (0.6, 0.4, 0.8)
#notes whose starts are this close (in ticks) are one chord
clusterTicks = 48


#the reads run in separate processes: most of a read is Python code, which holds the interpreter's lock, so reads on threads take turns (three
#reads together took 62 seconds where one takes 36). each process keeps its own copy of the templates and the page. False reads on threads again
useProcesses = True
#threads each of the reads' OpenCV calls may use (three reads at once share the machine's cores)
cvThreads = 3
_pool = None
_poolLock = threading.Lock()


def _work(path, blur, wantTrace):
    #runs in a worker process: one read, with its bar warnings and (for the first read) its trace sent back
    import cv2
    cv2.setNumThreads(cvThreads)
    trace = {} if wantTrace else None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = readScore(path, trace, None, not wantTrace, blur)
    return result, trace, [str(warning.message) for warning in caught]


def _getPool(size):
    global _pool
    with _poolLock:
        if _pool is None:
            _pool = ProcessPoolExecutor(max_workers=size)
        return _pool


def warmPool(size=3):
    #start the worker processes before the first scan needs them (importing OpenCV and the reader takes a few seconds in each)
    if not useProcesses:
        return
    try:
        pool = _getPool(size)
        for _ in range(size):
            pool.submit(_ping)
    except Exception:
        pass


def _ping():
    import cv2  # noqa: F401
    from . import score  # noqa: F401
    return True


def _readInProcesses(amounts, path, trace, traced=True):
    pool = _getPool(len(amounts))
    futures = [pool.submit(_work, path, amount, traced and index == 0) for index, amount in enumerate(amounts)]
    reads = []
    for index, future in enumerate(futures):
        try:
            result, childTrace, messages = future.result()
        except Exception as error:
            if index == 0 and traced:
                raise
            probe(trace, "vote", "skipped", blur=amounts[index], error=f"{type(error).__name__}: {error}")
            continue
        if index == 0 and traced:
            if trace is not None and childTrace:
                trace.update(childTrace)
            #the first read's bar warnings are the page's: raised here so whoever is catching them (the scanner) sees them
            for message in messages:
                warnings.warn(message)
        reads.append(result)
    return reads


def readVoted(path, trace=None, stage="all"):
    #stage "first": only the normal read. "rest": only the other reads of the vote (to add to the first when it did not decode on its own).
    #"all": every read, as before
    #[(notes, ticksPerBeat), ...], the normal read first. with the vote switched on the page is read at three amounts of deblurring (an
    #amount that is right for one photo is too little or too much for another, and a chord wrong in one read is usually right in the
    #others); the decoder then takes a vote on every chord. the first read runs here (so its bar warnings are caught by the caller) and
    #the others on worker threads, quietly. a read that fails is skipped
    amounts = voteBlurs if features.on("vote") else (None,)
    traced = stage != "rest"
    if stage == "first":
        amounts = amounts[:1]
    elif stage == "rest":
        amounts = amounts[1:]
    if not amounts:
        return []
    if useProcesses and len(amounts) > 1:
        try:
            return _readInProcesses(amounts, path, trace, traced)
        except ReaderError:
            raise
        except Exception as error:
            #the processes could not be used (a pool that broke, no memory): read on threads instead
            global _pool
            _pool = None
            probe(trace, "vote", "processes failed, using threads", error=f"{type(error).__name__}: {error}")
    with ThreadPoolExecutor(max_workers=max(1, len(amounts) - 1)) as pool:
        others = [pool.submit(readScore, path, None, None, True, amount) for amount in amounts[1:]]
        reads = [readScore(path, trace if traced else None, None, not traced, amounts[0])]
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
