import resvg_py
import verovio

from .errors import RendererError


def musicXmlToSvg(musicXml, pageWidth):
    toolkit = verovio.toolkit()
    toolkit.setOptions({
        "pageWidth": pageWidth,
        "adjustPageHeight": True,
        "scale": 60,
        "header": "none",
        "footer": "none",
    })
    if not toolkit.loadData(musicXml):
        raise RendererError("verovio could not load the score")
    return toolkit.renderToSVG(1)


def svgToPng(svg, zoom):
    return bytes(resvg_py.svg_to_bytes(svg_string=svg, background="#ffffff", zoom=zoom))
