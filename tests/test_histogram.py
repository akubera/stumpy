#
# test/histogram.py
#

import pytest
import numpy as np
from stumpy.histogram import Histogram


@pytest.fixture
def ROOT():
    return pytest.importorskip("ROOT")


@pytest.fixture
def h1f(ROOT):
    np.random.seed(45)
    h1 = ROOT.TH1F("h", "hist title", 100, 0.0, 1.0)
    for num in np.random.random(1000):
        h1.Fill(num)
    tuple(h1.Fill(5) for i in range(5))
    tuple(h1.Fill(-5) for i in range(5))
    return h1


@pytest.fixture
def h2f(ROOT):
    np.random.seed(45)
    h2 = ROOT.TH2F("h", "hist title",
                   100, 0.0, 1.0,
                   100, 0.0, 1.0)
    for x, y in np.random.random((1000, 2)):
        h2.Fill(x, y)
    return h2


@pytest.fixture
def h3(ROOT):
    h3 = ROOT.TH3F("h", "hist title",
                   100, 0.0, 1.0,
                   100, 0.0, 1.0,
                   100, 0.0, 1.0)
    return h3


def test_histogram_constructor_ROOT_TH1(h1f):
    hist = Histogram.BuildFromRootHist(h1f)
    print(hist.data)
    assert hist.shape == (100, )
    assert hist.name == h1f.GetName()
    assert hist.title == h1f.GetTitle()
    assert hist.underflow == h1f.GetBinContent(0)
    assert hist.overflow == h1f.GetBinContent(101)


def notest_histogram_constructor_ROOT_TH2(h2f):
    hist = Histogram.BuildFromRootHist(h2f)
    print(hist.data)
    assert hist.shape is (100, 100, )


def test_histogram_constructor():
	np.random.seed(42)
	data = np.random.random(45)
	axis = np.linspace(0.0, 1.0, num=45)
	print(axis)
	hist = Histogram(data, axis)
	assert len(hist.axes) is 1

	x_axis = hist.axes[0]
	assert len(x_axis._bin_centers) is 45
	assert x_axis[2] == 45/990
