#turns a photo or scan of a printed score into a clean, straight, black-on-white image for the reader
#simplest methods first: find the page, flatten the lighting, threshold, straighten, despeckle
#   python -m reader.rectify --input photo.jpg --output clean.png
import argparse
import sys

import cv2
import numpy

#photos are worked on at about this long side, so the blur and kernel sizes below are sensible
workingSide = 2400


def loadColour(imagePath):
    image = cv2.imread(str(imagePath), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not read image: " + str(imagePath))
    return image


def resizeLongSide(image, side):
    factor = side / max(image.shape[:2])
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, None, fx=factor, fy=factor, interpolation=interpolation)


def orderCorners(points):
    #top left, top right, bottom right, bottom left
    points = points.reshape(4, 2).astype(numpy.float32)
    total, difference = points.sum(axis=1), numpy.diff(points, axis=1).ravel()
    return numpy.array([
        points[numpy.argmin(total)], points[numpy.argmin(difference)],
        points[numpy.argmax(total)], points[numpy.argmax(difference)],
    ])


def findPage(gray):
    #the page (or the window on a screen) is the biggest bright patch: smear the notes away, threshold,
    #take the largest bright blob, preferring one that does not run off the photo (a white desk does),
    #and fit four corners to it; None if nothing big enough
    height, width = gray.shape
    size = max(gray.shape) // 60 | 1
    smeared = cv2.GaussianBlur(cv2.morphologyEx(gray, cv2.MORPH_CLOSE, numpy.ones((size, size), numpy.uint8)), (size, size), 0)
    _, bright = cv2.threshold(smeared, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(bright, connectivity=4)
    blobs = []
    for label in range(1, count):
        left, top, blobWidth, blobHeight, area = stats[label]
        if area < 0.1 * gray.size:
            continue
        touchesEdge = left == 0 or top == 0 or left + blobWidth == width or top + blobHeight == height
        blobs.append((touchesEdge, -area, label))
    if not blobs:
        return None
    label = min(blobs)[2]
    contours, _ = cv2.findContours((labels == label).astype(numpy.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    for tolerance in (0.01, 0.02, 0.04, 0.08):
        corners = cv2.approxPolyDP(hull, tolerance * cv2.arcLength(hull, True), True)
        if len(corners) == 4:
            return orderCorners(corners)
    return orderCorners(cv2.boxPoints(cv2.minAreaRect(hull)))


def warpPage(image, corners):
    topLeft, topRight, bottomRight, bottomLeft = corners
    width = int(max(numpy.linalg.norm(topRight - topLeft), numpy.linalg.norm(bottomRight - bottomLeft)))
    height = int(max(numpy.linalg.norm(bottomLeft - topLeft), numpy.linalg.norm(bottomRight - topRight)))
    target = numpy.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], numpy.float32)
    return cv2.warpPerspective(
        image, cv2.getPerspectiveTransform(corners, target), (width, height), borderMode=cv2.BORDER_REPLICATE
    )


def flattenLighting(gray):
    #divide by a blurred copy, so shadows and uneven light become plain white paper
    size = max(gray.shape) // 20 | 1
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, numpy.ones((size // 4 | 1, size // 4 | 1), numpy.uint8))
    background = cv2.GaussianBlur(background, (size, size), 0)
    return cv2.divide(gray, background, scale=255)


def binarise(gray):
    #after flattening, paper is near white: anything clearly darker is ink, even a faint blurred staff line.
    #otsu alone sits halfway to the black of the noteheads and loses those lines
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary = cv2.threshold(gray, max(otsu, 200), 255, cv2.THRESH_BINARY)
    return binary


def skewAngle(binary, searchDegrees=10):
    #the angle that makes the rows of ink sharpest: staff lines line up with the rows when level
    ink = resizeLongSide(255 - binary, 800)
    centre = (ink.shape[1] / 2, ink.shape[0] / 2)

    def sharpness(angle):
        turn = cv2.getRotationMatrix2D(centre, angle, 1.0)
        rotated = cv2.warpAffine(ink, turn, (ink.shape[1], ink.shape[0]), flags=cv2.INTER_NEAREST)
        return float(numpy.var(rotated.sum(axis=1, dtype=numpy.float64)))

    best = max(numpy.arange(-searchDegrees, searchDegrees + 0.01, 0.5), key=sharpness)
    return max(numpy.arange(best - 0.5, best + 0.51, 0.05), key=sharpness)


def straightenLines(binary, gray):
    #one perspective warp that makes long lines level (staff lines) or upright (barlines):
    #each line's ends are asked to move onto a level or upright line through its middle
    ink = 255 - binary
    height, width = binary.shape
    segments = []
    for minimum, isLevel in ((0.25 * width, True), (0.06 * height, False)):
        found = cv2.HoughLinesP(ink, 1, numpy.pi / 720, 100, minLineLength=int(minimum), maxLineGap=int(0.005 * width) + 2)
        for x1, y1, x2, y2 in ([] if found is None else found.reshape(-1, 4)):
            angle = abs(numpy.degrees(numpy.arctan2(y2 - y1, x2 - x1))) % 180
            if isLevel and min(angle, 180 - angle) < 15:
                segments.append(((x1, y1), (x2, y2), ((x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2))))
            elif not isLevel and abs(angle - 90) < 15:
                segments.append(((x1, y1), (x2, y2), (((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2))))
    if len(segments) < 4:
        return binary, gray
    source = numpy.float32([point for start, end, _ in segments for point in (start, end)])
    target = numpy.float32([point for _, _, targets in segments for point in targets])
    homography, _ = cv2.findHomography(source, target, cv2.RANSAC, 0.003 * width + 1)
    if homography is None:
        return binary, gray
    #the same warp for the grey picture the reader will get
    return tuple(
        cv2.warpPerspective(image, homography, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
        for image in (binary, gray)
    )


def rotate(binary, angle):
    centre = (binary.shape[1] / 2, binary.shape[0] / 2)
    turn = cv2.getRotationMatrix2D(centre, angle, 1.0)
    return cv2.warpAffine(
        binary, turn, (binary.shape[1], binary.shape[0]),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


def despeckle(binary, smallest):
    #drop specks of ink smaller than a staff-line dot, keep everything else
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    keep = numpy.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= smallest
    cleaned = numpy.full_like(binary, 255)
    cleaned[keep[labels]] = 0
    return cleaned


def removeClutter(binary):
    #a score is thin lines and small marks: drop ink touching the edge of the photo (desk, bezel)
    #and big solid patches (dark bars, text blocks), keep the rest
    count, labels, stats, _ = cv2.connectedComponentsWithStats(255 - binary, connectivity=8)
    height, width = binary.shape
    keep = numpy.ones(count, bool)
    keep[0] = False
    for label in range(1, count):
        left, top, boxWidth, boxHeight, area = stats[label]
        touchesEdge = left == 0 or top == 0 or left + boxWidth == width or top + boxHeight == height
        isSolid = area > 0.002 * binary.size and area > 0.3 * boxWidth * boxHeight
        #a whole line of music is one blob too (staff lines, brace, barlines and every note touching them), and
        #in a slightly soft photo its thick lines can make it 30% ink or more. it is never solid, though: it has
        #staff lines, separate bands of ink running across it with paper between them
        if isSolid and hasStaffLines(labels[top:top + boxHeight, left:left + boxWidth] == label):
            isSolid = False
        keep[label] = not isSolid
    cleaned = numpy.full_like(binary, 255)
    cleaned[keep[labels]] = 0
    return cleaned


def hasStaffLines(blob, minimumBands=5, coverage=0.6):
    #rows where ink runs across most of the blob, grouped into bands; a staff gives five per staff,
    #a dark bar or a bezel gives one solid band
    rows = blob.mean(axis=1) >= coverage
    bands = int(numpy.count_nonzero(numpy.diff(rows.astype(numpy.int8)) == 1)) + int(rows[0])
    return bands >= minimumBands and rows.mean() < 0.5


def trimMargins(binary, margin=20):
    ink = numpy.argwhere(binary < 128)
    if len(ink) == 0:
        return binary
    (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
    cropped = binary[top:bottom + 1, left:right + 1]
    return cv2.copyMakeBorder(cropped, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=255)


def markerCandidates(binary):
    #every black square holding a white square holding a black one: (centre, average outer width).
    #a corner of the page tilted away from the camera squashes its marker into a flat diamond, so the
    #width and height may differ a lot; the three nested squares are what makes it a marker
    contours, hierarchy = cv2.findContours(255 - binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]
    found = []
    for index, contour in enumerate(contours):
        child = hierarchy[index][2]
        if child < 0 or hierarchy[child][2] < 0:
            continue
        grandchild = hierarchy[child][2]
        outer, hole, centre = (cv2.contourArea(contours[i]) for i in (index, child, grandchild))
        if outer < 50 or not (0.3 < hole / outer < 0.75 and 0.08 < centre / outer < 0.35):
            continue
        left, top, width, height = cv2.boundingRect(contour)
        if not 0.25 < width / height < 4:
            continue
        moments = cv2.moments(contours[grandchild])
        if moments["m00"] == 0:
            continue
        found.append((numpy.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]]), (width + height) / 2))
    return found


def mergeCandidates(groups):
    #the same marker found in several thresholdings is one marker
    merged = []
    for group in groups:
        for centre, size in group:
            if not any(numpy.linalg.norm(centre - other) < 0.5 * max(size, otherSize) for other, otherSize in merged):
                merged.append((centre, size))
    return merged


def pickMarkers(found, imageShape):
    #the three corner markers of the page: other square-in-square shapes in the photo (an app icon on a phone,
    #a key on a keyboard) are told apart by where they are. the three must make a right angle, with sides that
    #span a good part of the photo, and be alike in size. of the trios that do, the widest wins.
    #returns (topLeft, topRight, bottomLeft, size) or None
    from itertools import combinations

    shortest = 0.2 * min(imageShape[:2])
    best = None
    for trio in combinations(found[:12], 3):
        sizes = [size for _, size in trio]
        if max(sizes) / min(sizes) > 2.0:
            continue
        centres = [centre for centre, _ in trio]
        #the corner opposite the longest side is the top left one; the right angle is there
        pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
        first, second, corner = max(pairs, key=lambda pair: numpy.linalg.norm(centres[pair[0]] - centres[pair[1]]))
        topLeft, one, other = centres[corner], centres[first], centres[second]
        a, b = one - topLeft, other - topLeft
        lengthA, lengthB = numpy.linalg.norm(a), numpy.linalg.norm(b)
        if min(lengthA, lengthB) < shortest:
            continue
        angle = numpy.degrees(numpy.arccos(numpy.clip(numpy.dot(a, b) / (lengthA * lengthB), -1, 1)))
        if abs(angle - 90) > 25:
            continue
        span = lengthA * lengthB
        if best is None or span > best[0]:
            #going clockwise from top left on screen (y down) reaches top right before bottom left
            topRight, bottomLeft = (one, other) if a[0] * b[1] - a[1] * b[0] > 0 else (other, one)
            best = (span, (topLeft, topRight, bottomLeft, float(numpy.mean(sizes))))
    return None if best is None else best[1]


def findMarkers(binary):
    #the renderer's three corner finder squares, found in one black and white picture
    return pickMarkers(markerCandidates(binary), binary.shape)


def warpByMarkers(gray, markers):
    #map the marker centres onto an upright rectangle, the fourth corner completing the parallelogram,
    #then paint over the markers so the reader never sees them
    topLeft, topRight, bottomLeft, size = markers
    bottomRight = topRight + bottomLeft - topLeft
    width = numpy.linalg.norm(topRight - topLeft)
    height = numpy.linalg.norm(bottomLeft - topLeft)
    #work at full resolution whatever the photo's size; stay inside the page, whose edge is just past the markers
    scale = workingSide / (width + size)
    width, height, size = int(width * scale), int(height * scale), size * scale
    #the fourth corner is a guess: three markers only fix a parallelogram, and a photo taken from a little
    #below or to the side makes the far side of the page wider than that. a curled page also bulges past the
    #straight line between two markers. so keep a margin on every side, or the clefs and the ends of the lines
    #of music get painted out; straightenLines and straightenStaves later remove what is left of the bend
    slack = int(round(max(2 * size, 0.06 * width)))
    inset = int(round(0.5 * size)) + slack
    source = numpy.float32([topLeft, topRight, bottomRight, bottomLeft])
    target = numpy.float32([[inset, inset], [inset + width, inset], [inset + width, inset + height], [inset, inset + height]])
    warped = cv2.warpPerspective(
        gray, cv2.getPerspectiveTransform(source, target), (width + 2 * inset, height + 2 * inset),
        borderMode=cv2.BORDER_REPLICATE,
    )
    #the music sits inside the box the marker centres make, give or take the margin; outside it are only the
    #page edges and the desk
    edge = inset - slack
    warped[:edge, :] = 255
    warped[inset + height + slack:, :] = 255
    warped[:, :edge] = 255
    warped[:, inset + width + slack:] = 255
    half = int(0.75 * size)
    for x, y in ((inset, inset), (inset + width, inset), (inset, inset + height)):
        warped[max(0, int(y) - half):int(y) + half, max(0, int(x) - half):int(x) + half] = 255
    return warped


def rectify(imagePath, steps=None):
    #returns a grayscale image (0 ink, 255 paper); steps, if a dict, collects each stage for debugging
    steps = {} if steps is None else steps
    colour = resizeLongSide(loadColour(imagePath), workingSide)
    gray = cv2.cvtColor(colour, cv2.COLOR_BGR2GRAY)
    #a photo of a screen shows the pixel grid as fine dots: blur it away before thresholding
    unblur = max(1.0, max(gray.shape) / 1200)
    markers = findMarkersInGray(gray)
    corners = None
    if markers is not None:
        gray = warpByMarkers(gray, markers)
    else:
        corners = findPage(gray)
        if corners is not None:
            gray = warpPage(gray, corners)
    angle = cleanPage(gray, markers is not None, steps)
    found = "markers" if markers is not None else ("page edge" if corners is not None else "nothing")
    steps["info"] = f"found by {found}, rotated by {angle:.2f} degrees"
    return steps["result"]


def findMarkersInGray(gray):
    #the corner markers in a raw grey photo. a soft photo turns a marker's thin white ring grey and one fixed
    #threshold fills it in; a squashed or shaded corner can need another. so markers are collected from the
    #blurred and sharp pictures and from thresholds that compare each pixel with its neighbourhood, and the
    #three corners are chosen from all of them together
    unblur = max(1.0, max(gray.shape) / 1200)
    groups = [
        markerCandidates(binarise(flattenLighting(cv2.GaussianBlur(gray, (0, 0), unblur)))),
        markerCandidates(binarise(flattenLighting(gray))),
    ]
    for block in (31, 51, 91):
        for offset in (3, 8, 15):
            groups.append(markerCandidates(
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, offset)
            ))
    return pickMarkers(onPaper(mergeCandidates(groups), gray), gray.shape)


def onPaper(found, gray):
    #a marker is printed on the white page, so the ring just outside it is paper. an app icon on a phone or a
    #key on a keyboard sits on something dark: drop candidates whose surroundings are much darker than the page
    paper = float(numpy.percentile(gray, 95))
    height, width = gray.shape
    rows, columns = numpy.ogrid[:height, :width]
    kept = []
    for centre, size in found:
        x, y = centre
        reach = int(1.3 * size) + 2
        top, bottom = max(0, int(y) - reach), min(height, int(y) + reach + 1)
        left, right = max(0, int(x) - reach), min(width, int(x) + reach + 1)
        distance = numpy.hypot(columns[:, left:right] - x, rows[top:bottom] - y)
        ring = gray[top:bottom, left:right][(distance > 0.8 * size) & (distance < 1.3 * size)]
        if len(ring) and numpy.median(ring) >= 0.7 * paper:
            kept.append((centre, size))
    return kept


def cleanPage(gray, hasMarkers, steps):
    #everything after the page warp: gray is the upright (or roughly cropped) page.
    #fills steps["page"] .. steps["result"] and returns the skew angle it removed
    unblur = max(1.0, max(gray.shape) / 1200)
    markers = True if hasMarkers else None
    steps["page"] = gray
    flat = flattenLighting(cv2.GaussianBlur(gray, (0, 0), unblur))
    steps["flat"] = flat
    binary = binarise(flat)
    if markers is None:
        #staff lines run across the score: if the ink lines up better in columns, the score is on its side
        ink = 255.0 - resizeLongSide(binary, 800)
        if numpy.var(ink.sum(axis=0)) > numpy.var(ink.sum(axis=1)):
            binary = cv2.rotate(binary, cv2.ROTATE_90_CLOCKWISE)
            flat = cv2.rotate(flat, cv2.ROTATE_90_CLOCKWISE)
    steps["binary"] = binary
    angle = skewAngle(binary, searchDegrees=2 if markers is not None else 10)
    straight, flat = rotate(binary, angle), rotate(flat, angle)
    for _ in range(2):
        straight, flat = straightenLines(binarise(straight), flat)
    straight = binarise(straight)
    steps["straight"] = straight
    kept = removeClutter(despeckle(straight, smallest=max(4, straight.size // 2_000_000)))
    #the reader gets the smooth grey picture, which matches its templates far better than hard black and white;
    #only the ink the clean-up threw away (specks, clutter) is painted out
    removed = cv2.dilate(((straight < 128) & (kept > 128)).astype(numpy.uint8), numpy.ones((3, 3), numpy.uint8))
    grey = flat.copy()
    grey[removed > 0] = 255
    ink = numpy.argwhere(kept < 128)
    if len(ink):
        (top, left), (bottom, right) = ink.min(axis=0), ink.max(axis=0)
        grey = grey[top:bottom + 1, left:right + 1]
    result = cv2.copyMakeBorder(grey, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    #a page held up or lying on a curve bows: bend the staves level so each staff line is one height
    from .dewarp import straightenStaves
    result = straightenStaves(result)
    steps["result"] = result
    return angle


def main():
    parser = argparse.ArgumentParser(prog="reader.rectify", description="clean up a photo of a score for the reader")
    parser.add_argument("--input", required=True, help="photo or scan")
    parser.add_argument("--output", required=True, help="clean .png to write")
    parser.add_argument("--steps", help="folder to write every stage into, for debugging")
    arguments = parser.parse_args()
    steps = {}
    try:
        result = rectify(arguments.input, steps)
    except ValueError as error:
        print("error: " + str(error), file=sys.stderr)
        sys.exit(1)
    cv2.imwrite(arguments.output, result)
    if arguments.steps:
        for index, name in enumerate(("page", "flat", "binary", "straight", "result")):
            cv2.imwrite(f"{arguments.steps}/{index}-{name}.png", steps[name])
    print(f"wrote {arguments.output} ({steps['info']})")


if __name__ == "__main__":
    main()
