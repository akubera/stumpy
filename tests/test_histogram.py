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
def rnd_seed():
    return 45


@pytest.fixture
def h1f(ROOT, rnd_seed):
    np.random.seed(rnd_seed)
    h1 = ROOT.TH1F("h1f", "hist title", 100, 0.0, 1.0)
    for num in np.random.random(1000):
        h1.Fill(num)
    tuple(h1.Fill(5) for i in range(100))
    tuple(h1.Fill(-5) for i in range(106))
    print('::', [h1.GetBinContent(i) for i in range(20)], end=' ... ')
    print([h1.GetBinContent(i) for i in range(80, 103)])
    return h1


@pytest.fixture
def h2f(ROOT, rnd_seed):
    np.random.seed(rnd_seed)
    h2 = ROOT.TH2F("h2f", "hist title",
                   100, 0.0, 1.0,
                   70, 0.0, 1.0)
    for x, y in np.random.random((50000, 2)):
        h2.Fill(x, y)

    for v in np.random.random(1000):
        h2.Fill(-1, v)
    for v in np.random.random(1000):
        h2.Fill(v, -1)
    for v in np.random.random(1000):
        h2.Fill(2, v)
    for v in np.random.random(1000):
        h2.Fill(v, 2)

    return h2


@pytest.fixture
def h3(ROOT):
    h3 = ROOT.TH3F("h3f", "hist title",
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
    for i, x in enumerate(hist.data):
        assert x == h1f.GetBinContent(i+1)


def test_histogram_constructor_ROOT_TH2_simple(ROOT):
    h2f = ROOT.TH2F("h2f_simple", "hist title",
                   7, 0.0, 1.0,
                   3, 0.0, 1.0)
    h2f.Fill(0.1, 0.1, 2.0)   # x_bin = 0, y_bin = 0
    h2f.Fill(0.1, 0.5, 3.0)   # x_bin = 0, y_bin = 1
    h2f.Fill(0.1, 0.8, 4.0)   # x_bin = 0, y_bin = 2

    h2f.Fill(0.2, 0.1, 3.0)   # x_bin = 1, y_bin = 0
    h2f.Fill(0.2, 0.5, 9.0)   # x_bin = 1, y_bin = 1
    h2f.Fill(0.2, 0.8, 7.0)   # x_bin = 1, y_bin = 2

    h2f.Fill(0.3, 0.5, 12.0)  # x_bin = 2, y_bin = 0
    h2f.Fill(0.5, 0.1, 3.0)   # x_bin = 3, y_bin = 0
    h2f.Fill(0.6, 0.9, 13.0)  # x_bin = 4, y_bin = 0
    h2f.Fill(0.8, 0.5, 9.0)   # x_bin = 5, y_bin = 1
    h2f.Fill(0.8, 0.8, 14.0)  # x_bin = 5, y_bin = 2
    h2f.Fill(0.9, 0.8, 7.0)   # x_bin = 6, y_bin = 2

    h2f.Fill(-1, 0.1, 100.0)   # x_bin = underflow, y_bin = 0
    h2f.Fill(-1, 0.5, 150.0)   # x_bin = underflow, y_bin = 1
    h2f.Fill(2., 0.5, 200.0)   # x_bin = overflow, y_bin = 1

    h2f.Fill(0.1, -1, 300.0)   # x_bin = 0, y_bin = underflow
    h2f.Fill(0.6, -1, 350.0)   # x_bin = 4, y_bin = underflow
    h2f.Fill(0.3, 1.2, 400.0)  # x_bin = 2, y_bin = overflow
    h2f.Fill(0.9, 1.2, 450.0)  # x_bin = 6, y_bin = overflow

    hist = Histogram.BuildFromRootHist(h2f)
    assert hist.shape == (h2f.GetNbinsX(), h2f.GetNbinsY())

    x_underflow, y_underflow, = hist.underflow
    x_overflow, y_overflow = hist.overflow
    assert x_overflow.shape == x_underflow.shape
    assert y_overflow.shape == y_underflow.shape

    assert all(x_underflow == [0.0, 100.0, 150.0, 0.0, 0.0])
    assert all(x_overflow == [0.0, 0.0, 200.0, 0.0, 0.0])

    assert all(y_underflow == [0.0, 300.0,  0.0, 0.0, 0.0, 350.0, 0.0, 0.0, 0.0])
    assert all(y_overflow == [0.0, 0.0, 0.0, 400.0, 0.0, 0.0, 0.0, 450.0, 0.0])

    for y_bin in range(3 + 2):
        assert h2f.GetBinContent(0, y_bin) == hist.underflow[0][y_bin]
        assert h2f.GetBinContent(8, y_bin) == hist.overflow[0][y_bin]

    for x_bin in range(7 + 2):
        assert h2f.GetBinContent(x_bin, 0) == hist.underflow[1][x_bin]
        assert h2f.GetBinContent(x_bin, 4) == hist.overflow[1][x_bin]

    for x, xs in enumerate(hist.data):
        for y, v in enumerate(xs):
            assert v == h2f.GetBinContent(x + 1, y + 1)


@pytest.mark.parametrize('rnd_seed', [100, 45, 0, 0])
def test_histogram_constructor_ROOT_TH2(h2f):
    hist = Histogram.BuildFromRootHist(h2f)
    assert hist.shape == (100, 70)

    for i in range(70 + 2):
        assert h2f.GetBinContent(0, i) == hist.underflow[0][i]
        assert h2f.GetBinContent(101, i) == hist.overflow[0][i]

    for i in range(100 + 2):
        assert h2f.GetBinContent(i, 0) == hist.underflow[1][i]
        assert h2f.GetBinContent(i, 71) == hist.overflow[1][i]

    for x, xs in enumerate(hist.data):
        for y, v in enumerate(xs):
            assert v == h2f.GetBinContent(x + 1, y + 1)



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
