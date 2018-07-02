#
# stumpy/jupyter.py
#
"""
Methods for running in a jupyter notebook
"""

from .histogram import Histogram, HistogramRatioPair
from . import utils
import numpy as np


def enable_inline_ROOT_stuff():
    from IPython.display import Javascript, display
    display(Javascript("""
    console.log("enabling JSRoot")
    requirejs.config({
    paths: {JSRoot: [
        '/files/post_analysis/jsroot/scripts/JSRootCore',
        'https://raw.githubusercontent.com/linev/jsroot/master/scripts/JSRootCore'
    ]}
    });
    require(["JSRoot"], function (ROOT) {
        console.log("[stumpy::enable_inline_ROOT_stuff] Loaded JSRoot", ROOT);
    })
    """))


def draw(obj, drawops='', size=None):
    """
    
    """
    c = TCanvas()
    if size is not None:
        c.SetCanvasSize(size)
    obj.Draw(drawops)
    c.Draw()
    return c

