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


def get_root_object(obj, paths):
    """
    Return a root object contained in the obj
    """
    if isinstance(paths, (list, tuple)):
        if len(paths):
            path, paths = paths[0], paths[1:]
        else:
            return None
    else:
        path, paths = paths, []

    key, *rest = path.split('.', 1)
    try:
        new_obj = obj.Get(key)
    except AttributeError:
        new_obj = obj.FindObject(key)

    if new_obj == None and len(paths) is not 0:
        return get_root_object(obj, paths)
    elif new_obj == None or len(rest) is 0:
        return new_obj
    else:
        return get_root_object(new_obj, rest[0])
