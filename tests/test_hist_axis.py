#
# test/histogram.py
#

import pytest
import numpy as np
import itertools
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
