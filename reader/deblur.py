#a photo of a page is blurrier than the renderer's own picture, and the reader's symbol templates are cut from the crisp render, so on a
#blurred page the matches weaken and the choices between near equal candidates turn into guesses. the standard answer for a known
#blur is deconvolution: estimate how wide the blur is and undo it (Richardson-Lucy). how wide is read off the staff lines themselves:
#a line is a one-pixel-thin stroke, so how wide it looks across is the blur.
#   sharper, sigma = deblur(gray, staves, spacing)
import cv2
import numpy

from . import features
from .probe import probe

#what a staff line looks like across on a crisp page, as a standard deviation, in staff spaces
crispLine = 0.05
#a blur narrower than this (pixels) is left alone: the page is crisp
leaveAlone = 1.0
#more than this is not believed
widest = 5.0
#the measured width over-reads (a line is not infinitely thin, and the page was stretched), so only this share of it is undone: on the
#phone photo the measured width was 4.9 pixels and the width that worked was between 2 and 3
strength = 0.5
#the vote's normal share (reader/vote.py): the other reads are scaled around the adaptive amount by their share over this one
voteCentre = 0.6
#how much thinner than its blurred self a line must get before the search stops: a crisp line is this many times its crisp width, at most
closeEnough = 1.25
#blur widths tried (pixels, standard deviation of the Gaussian), smallest first
#the stretch of a line (in staff spaces) the adaptive search measures over
shortStretch = 0.5
trials = (1.0, 1.6, 2.2, 2.8, 3.4, 4.0, 4.6, 5.2, 6.0)


def lineWidth(gray, staves, spacing, stretchSpaces=3.0):
    #the median full width at half height of a staff line's dark band, over many stretches of many lines (pixels), or None. a stretch is
    #long enough to average out noise but a line that wanders across it (a wavy page) looks wider than it is: the adaptive search measures
    #over short stretches, where a wavy line is nearly straight
    ink = 255.0 - gray.astype(numpy.float32)
    stretch = max(8, int(stretchSpaces * spacing))
    widths = []
    for staff in staves[:12]:
        left, right = int(staff.left), int(staff.right)
        for start in range(left, right - stretch, stretch):
            for line in staff.lines:
                low, high = int(round(line - 0.45 * spacing)), int(round(line + 0.45 * spacing))
                if low < 0 or high >= ink.shape[0]:
                    continue
                #the median across the stretch, taken only for the rows around this line (every row's median is its own, so this is the same
                #numbers as taking it for the whole height and cutting out the rows, a hundred times faster)
                segment = numpy.median(ink[low:high + 1, start:start + stretch], axis=1)
                peak = int(segment.argmax())
                base = float(numpy.percentile(segment, 20))
                height = float(segment[peak]) - base
                if height < 25:
                    continue
                half = base + height / 2
                a = peak
                while a > 0 and segment[a] > half:
                    a -= 1
                b = peak
                while b < len(segment) - 1 and segment[b] > half:
                    b += 1
                widths.append(float(b - a))
    return float(numpy.median(widths)) if len(widths) >= 20 else None


def richardsonLucy(gray, sigma, rounds=12):
    size = int(6 * sigma) | 1
    #the blur is a Gaussian, which is the same as two passes of a one-dimensional kernel (and the Gaussian is symmetric, so the flipped blur
    #is the same blur): far fewer multiplications than one pass of the whole square kernel
    observed = gray.astype(numpy.float32) / 255.0 + 1e-4
    estimate = observed.copy()
    for _ in range(rounds):
        ratio = observed / (cv2.GaussianBlur(estimate, (size, size), sigma) + 1e-6)
        estimate *= cv2.GaussianBlur(ratio, (size, size), sigma)
    return numpy.clip(estimate * 255.0, 0, 255).astype(numpy.uint8)


def wiener(gray, sigma, noise=0.01):
    #deconvolution in one pass: the page's spectrum is divided by the blur's, with `noise` stopping it from boosting the finest detail (which is
    #mostly noise) without limit. Richardson-Lucy needs a hundred rounds to undo a blur this wide (the staff lines on a phone photo are
    #12 pixels across at half height), this does it at once
    image = gray.astype(numpy.float64) / 255.0
    height, width = image.shape
    pad = int(4 * sigma) + 1
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    rows, columns = padded.shape
    fy = numpy.fft.fftfreq(rows)[:, None]
    fx = numpy.fft.rfftfreq(columns)[None, :]
    transfer = numpy.exp(-2 * numpy.pi ** 2 * sigma ** 2 * (fx ** 2 + fy ** 2))
    restored = numpy.fft.irfft2(numpy.fft.rfft2(padded) * transfer / (transfer ** 2 + noise), s=(rows, columns))
    return numpy.clip(restored[pad:pad + height, pad:pad + width] * 255.0, 0, 255).astype(numpy.uint8)


def adaptiveSigma(gray, staves, spacing, trace=None):
    #how much blur to undo, found by trying it: a strip of the page (the first system) is deblurred at growing widths and the staff lines
    #measured each time; the answer is the first width at which a line is as thin as a crisp render's (a crisp line, times a little slack).
    #deblurring any further only adds ringing and noise. None if the lines cannot be measured or the page is already crisp
    first = lineWidth(gray, staves, spacing, shortStretch)
    target = crispLine * spacing * 2.355 * closeEnough
    if first is None or first <= max(target, leaveAlone * 2.355):
        return None
    pair = staves[:2]
    top = max(0, int(pair[0].lines[0] - 2.5 * spacing))
    bottom = min(gray.shape[0], int(pair[-1].lines[-1] + 2.5 * spacing))
    strip = gray[top:bottom]
    tried = []
    chosen = None
    for sigma in trials:
        trial = gray.copy()
        trial[top:bottom] = wiener(strip, sigma)
        width = lineWidth(trial, pair, spacing, shortStretch)
        tried.append((sigma, None if width is None else round(width, 2)))
        if width is None:
            continue
        if width <= target:
            chosen = sigma
            break
    probe(trace, "deblur", "adaptive", measured=round(first, 2), target=round(target, 2), tried=tried, chosen=chosen)
    return chosen


def deblur(gray, staves, spacing, trace=None, share=None):
    #the picture with its blur undone, or unchanged if it is already crisp (or the blur cannot be measured). share: how much of the
    #measured blur to undo (the page's normal share is `strength`); the reader's vote reads the page at several shares
    if features.on("adaptiveBlur"):
        chosen = adaptiveSigma(gray, staves, spacing, trace)
        if chosen is not None:
            #the vote's other reads go a little under and over the amount that was found
            return wiener(gray, chosen * (1.0 if share is None else share / voteCentre))
        if lineWidth(gray, staves, spacing) is not None and trace is not None:
            probe(trace, "deblur", "adaptive found nothing to undo", sigma=0)
    width = lineWidth(gray, staves, spacing)
    if width is None:
        probe(trace, "deblur", "measured", width=None, sigma=0)
        return gray
    total = width / 2.355
    sigma = float(numpy.sqrt(max(0.0, total * total - (crispLine * spacing) ** 2)))
    probe(trace, "deblur", "measured", width=round(width, 2), sigma=round(sigma, 2))
    share = strength if share is None else share
    if sigma * share < leaveAlone:
        return gray
    return richardsonLucy(gray, min(sigma, widest) * share)
