#a photo of a page is blurrier than the renderer's own picture, and the reader's symbol templates are cut from the crisp render, so on a
#blurred page the matches weaken and the choices between near equal candidates turn into guesses. the standard answer for a known
#blur is deconvolution: estimate how wide the blur is and undo it (Richardson-Lucy). how wide is read off the staff lines themselves:
#a line is a one-pixel-thin stroke, so how wide it looks across is the blur.
#   sharper, sigma = deblur(gray, staves, spacing)
import cv2
import numpy

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


def lineWidth(gray, staves, spacing):
    #the median full width at half height of a staff line's dark band, over many stretches of many lines (pixels), or None
    ink = 255.0 - gray.astype(numpy.float32)
    stretch = int(3 * spacing)
    widths = []
    for staff in staves[:12]:
        left, right = int(staff.left), int(staff.right)
        for start in range(left, right - stretch, stretch):
            profile = numpy.median(ink[:, start:start + stretch], axis=1)
            for line in staff.lines:
                low, high = int(round(line - 0.45 * spacing)), int(round(line + 0.45 * spacing))
                if low < 0 or high >= len(profile):
                    continue
                segment = profile[low:high + 1]
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
    kernel = cv2.getGaussianKernel(size, sigma)
    psf = (kernel @ kernel.T).astype(numpy.float32)
    flipped = psf[::-1, ::-1].copy()
    observed = gray.astype(numpy.float32) / 255.0 + 1e-4
    estimate = observed.copy()
    for _ in range(rounds):
        ratio = observed / (cv2.filter2D(estimate, -1, psf) + 1e-6)
        estimate *= cv2.filter2D(ratio, -1, flipped)
    return numpy.clip(estimate * 255.0, 0, 255).astype(numpy.uint8)


def deblur(gray, staves, spacing, trace=None, share=None):
    #the picture with its blur undone, or unchanged if it is already crisp (or the blur cannot be measured). share: how much of the
    #measured blur to undo (the page's normal share is `strength`); the reader's vote reads the page at several shares
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
