#
# stumpy/utils.py
#
"""
Utility methods
"""


ROOT_TO_NUMPY_DTYPE = {
    'C': 'i1',
    'S': 'i2',
    'I': 'i4',
    'F': 'f4',
    'D': 'f8',
}


def iter_tobject(tobj, pattern=None):
    from ROOT import TDirectory, TCollection
    from fnmatch import fnmatch
    import re

    _Regex = type(re.compile(''))

    if pattern is None:
        passes = lambda _: True
    elif isinstance(pattern, str):
        passes = lambda name: fnmatch(name, pattern)
    elif isinstance(pattern, _Regex):
        passes = lambda name: pattern.matches(name) is not None
    else:
        passes = pattern

    if isinstance(tobj, TDirectory):
        for key in tobj.GetListOfKeys():
            if passes(key.GetName()):
                yield key.ReadObj()

    elif isinstance(tobj, TCollection):
        for obj in tobj:
            if passes(obj.GetName()):
                yield obj


def get_root_object(obj, paths, delim='/'):
    """
    Return a root object contained in the obj
    """
    if isinstance(paths, (list, tuple)):
        for path in paths:
            found = get_root_object(obj, path)
            if not is_null(found):
                break
        else:
            found = None

        return found

    # seperate by dot delimiter
    key, *rest = paths.split(delim, 1)

    try:
        found_obj = obj.Get(key)
    except AttributeError:
        found_obj = obj.FindObject(key)

    if len(rest) is 0 or is_null(found_obj):
        return found_obj
    else:
        return get_root_object(found_obj, rest[0], delim)


def drawable_to_image(image_filename, *objs, return_nbimage=False):
    """
    Save root object Drawn to image_filename.
    If return nbimage is True, a Jupyter notebook displayable image
    is returned.
    """
    from ROOT import TCanvas, TImage
    c = TCanvas()
    for obj in objs:
        if isinstance(obj, tuple):
            obj, opts = obj
            obj.Draw(opts)
        else:
            obj.Draw()

    if image_filename.endswith(".eps"):
        c.Print(image_filename, 'eps')
    else:
        img = TImage.Create()
        img.FromPad(c)
        img.WriteImage(image_filename)
    if return_nbimage:
        from IPython.display import Image
        return Image(image_filename)


def root_histogram_datatype(hist):
    import re
    for next_class in hist.__class__.__mro__:
        classname = next_class.__name__
        m = re.search('TH[1-3](?P<root_type>[CSIFD])', classname)
        if m:
            return ROOT_TO_NUMPY_DTYPE[m.group('root_type')]
    else:
        raise TypeError("Not a root histogram")


def enumerate_histogram(hist, start=1, *, with_errors=False):
    """
    Get each index and with_error
    """
    enumerate_hist = {1: enumerate_1d,
                      2: enumerate_2d,
                      3: enumerate_3d}[hist.ndim]

    if with_errors:
        hist_iter = enumerate_hist(hist.data, hist.errors, start=start)
    else:
        hist_iter = enumerate_hist(hist.data, start=start)
    yield from hist_iter


def enumerate_1d(*args, start=0):
    for i, a in enumerate(zip(*args), start):
        yield (i, ), a


def enumerate_2d(*args, start=0):
    for i, a, in enumerate(zip(*args), start):
        for j, b in enumerate(zip(*a), start):
            yield (i, j), b


def enumerate_3d(*args, start=0):
    for (i, j), a in enumerate_2d(*args, start=start):
        for k, b in enumerate(zip(*a), start):
            yield (i, j, k), b


def is_null(obj):
    return obj == None

