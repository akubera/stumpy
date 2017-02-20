#
# stumpy/__init__.py
#
"""
"""

from .histogram import Histogram, HistogramRatioPair


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
