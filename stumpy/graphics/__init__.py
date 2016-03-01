#
# stumpy/graphics/__init__.py
#


def enable_bokeh():
    from bokeh.io import output_notebook
    output_notebook()
