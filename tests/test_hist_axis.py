#
# test/histogram.py
#

import pytest
import numpy as np

from itertools import zip_longest
from decimal import Underflow, Overflow
from stumpy.histogram import Histogram

Axis = Histogram.Axis


def test_axis_constructor():
    pass


def test_constructor_linear_spacing():
    a = Axis.BuildWithLinearSpacing(30, 0, 2.0)
    assert isinstance(a, Axis)
    assert a._xmin == 0.0
    assert a._xmax == 2
    assert a._bin_width == 2 / 30
    assert len(a._low_edges) == 30


@pytest.mark.parametrize("buildargs, vals, expected", [
    (dict(nbins=5, min_x=0, max_x=1), # [0.0, 0.2, 0.4, 0.6, 0.8. 1.0)
     [0.1, 0.22, 0.2, 1.0, 0.0, -0.1, 0.5, 0.600000001, 0.7, 0.8, 0.9],
     [0, 1, 1, Overflow, 0, Underflow, 2, 3, 3, 4, 4]),
    (dict(nbins=20, min_x=-1, max_x=1),
     [0.1000000001, 2.5, -2.5, -1.0, 0.99],
     [11, Overflow, Underflow, 0, 19]),
])
def test_getbin(buildargs, vals, expected):
    a = Axis.BuildWithLinearSpacing(**buildargs)
    for val, ex in zip_longest(vals, expected):
        assert a.getbin(val) == ex
        assert a.bin_at(val) == ex
