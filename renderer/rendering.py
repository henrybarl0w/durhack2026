import re
from importlib.resources import files

import resvg_py
import verovio

from .errors import RendererError

#verovio margin option, and the same margin in svg units (ten times larger)
pageMarginOption = 100
pageMargin = pageMarginOption * 10
#the left margin is wider: the system brace bulges at the middle of the left edge, where a finder marker sits
leftMarginOption = 150
leftMargin = leftMarginOption * 10


def musicXmlToSvg(musicXml, pageWidth):
    verovio.setDefaultResourcePath(str(files("verovio") / "data"))
    toolkit = verovio.toolkit()
    toolkit.setOptions({
        "pageWidth": pageWidth,
        "adjustPageHeight": True,
        #verovio's tallest page, shrunk to fit the music, so a long piece is never cut off after page 1
        "pageHeight": 60000,
        "scale": 60,
        "header": "none",
        "footer": "none",
        #wide margins leave room for the corner finder markers
        "pageMarginTop": pageMarginOption,
        "pageMarginBottom": pageMarginOption,
        "pageMarginLeft": leftMarginOption,
        "pageMarginRight": pageMarginOption,
    })
    if not toolkit.loadData(musicXml):
        raise RendererError("verovio could not load the score")
    return addFinderMarkers(toolkit.renderToSVG(1))


def finderMarker(x, y, size, hollow=False):
    #a qr-code style finder square: 7 by 7 black, 5 by 5 white inside, 3 by 3 black in the middle.
    #the hollow one (bottom right) has a 1 by 1 white hole in the middle, so the page can be told apart from
    #its upside down self: the reader turns the page until the hollow marker is at the bottom right
    unit = size / 7
    marker = (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{size:.2f}" height="{size:.2f}" fill="black" />'
        f'<rect x="{x + unit:.2f}" y="{y + unit:.2f}" width="{5 * unit:.2f}" height="{5 * unit:.2f}" fill="white" />'
        f'<rect x="{x + 2 * unit:.2f}" y="{y + 2 * unit:.2f}" width="{3 * unit:.2f}" height="{3 * unit:.2f}" fill="black" />'
    )
    if hollow:
        marker += f'<rect x="{x + 3 * unit:.2f}" y="{y + 3 * unit:.2f}" width="{unit:.2f}" height="{unit:.2f}" fill="white" />'
    return marker


def addFinderMarkers(svg):
    #finder squares in the four corners of the page and halfway along each of its four edges, inside its margin,
    #so a photo of the score can be found, straightened and turned the right way up. the corners fix the
    #perspective; the halfway ones show how far each edge bows, so a curled page can be flattened
    match = re.match(r'<svg width="([\d.]+)px" height="([\d.]+)px"', svg)
    viewBox = re.search(r'class="definition-scale"[^>]*viewBox="0 0 ([\d.]+) ', svg)
    if not match or not viewBox:
        return svg
    width, height = float(match.group(1)), float(match.group(2))
    margin = pageMargin * width / float(viewBox.group(1))
    size = 0.7 * margin
    offset = (margin - size) / 2
    top = offset
    left = (leftMargin * width / float(viewBox.group(1)) - size) / 2
    right, bottom = width - offset - size, height - offset - size
    middleX, middleY = (width - size) / 2, (height - size) / 2
    positions = (
        (left, top), (right, top), (left, bottom),
        (middleX, top), (middleX, bottom), (left, middleY), (right, middleY),
    )
    markers = "".join(finderMarker(x, y, size) for x, y in positions)
    markers += finderMarker(right, bottom, size, hollow=True)
    closing = svg.rstrip().rfind("</svg>")
    return svg[:closing] + markers + "\n" + svg[closing:]


def svgToPng(svg, zoom):
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", zoom=zoom))
