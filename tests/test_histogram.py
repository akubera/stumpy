#
# test/test_histogram.py
#

import pytest
import itertools
import numpy as np
from stumpy.histogram import Histogram


@pytest.fixture
def ROOT():
    return pytest.importorskip("ROOT")


@pytest.fixture(scope="module",
                params=[True, False])
def use_sumw2(request):
    return request.param


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
    return h1


@pytest.fixture
def h2f(ROOT, rnd_seed, use_sumw2):
    np.random.seed(rnd_seed)
    h2 = ROOT.TH2F("h2f", "hist title",
                   100, 0.0, 1.0,
                   70, 0.0, 1.0)
    if use_sumw2:
        h2.Sumw2()
    for x, y in np.random.random((5000, 2)):
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
def h3f(ROOT, rnd_seed):
    np.random.seed(rnd_seed)
    h3 = ROOT.TH3F("h3f", "hist title",
                   4, 0.0, 1.0,
                   9, 0.0, 1.0,
                   13, 0.0, 1.0)
    for x, y, z in np.random.random((5000, 3)):
        h3.Fill(x, y, z)

    for y, z in np.random.random((5000, 2)):
        h3.Fill(-1, y, z)
    for x, z in np.random.random((5000, 2)):
        h3.Fill(x, -1, z)
    for x, y in np.random.random((5000, 2)):
        h3.Fill(x, y, -1)

    for y, z in np.random.random((5000, 2)):
        h3.Fill(2, y, z)
    for x, z in np.random.random((5000, 2)):
        h3.Fill(x, 2, z)
    for x, y in np.random.random((5000, 2)):
        h3.Fill(x, y, 2)

    return h3


def test_histogram_constructor_ROOT_TH1(h1f):
    hist = Histogram.BuildFromRootHist(h1f)
    assert hist.shape == (100, )
    assert hist.name == h1f.GetName()
    assert hist.title == h1f.GetTitle()
    assert hist.underflow == h1f.GetBinContent(0)
    assert hist.overflow == h1f.GetBinContent(101)
    for i, x in enumerate(hist.data):
        assert x == h1f.GetBinContent(i+1)
    h1f.Delete()


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


@pytest.mark.parametrize('rnd_seed', [45, 100])
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
    h2f.Delete()


@pytest.mark.parametrize('rnd_seed', [42, ])
def test_histogram_constructor_ROOT_TH3(h3f):
    hist = Histogram.BuildFromRootHist(h3f)
    assert hist.shape == (4, 9, 13)

    for j, k in itertools.product(range(9 + 2), range(13 + 2)):
        assert h3f.GetBinContent(0, j, k) == hist.underflow[0][j, k]
        assert h3f.GetBinContent(5, j, k) == hist.overflow[0][j, k]

    for i, k in itertools.product(range(4 + 2), range(13 + 2)):
        assert h3f.GetBinContent(i, 0, k) == hist.underflow[1][i, k]
        assert h3f.GetBinContent(i, 10, k) == hist.overflow[1][i, k]

    for i, j in itertools.product(range(4 + 2), range(9 + 2)):
        assert h3f.GetBinContent(i, j, 0) == hist.underflow[2][i, j]
        assert h3f.GetBinContent(i, j, 14) == hist.overflow[2][i, j]

    for x, (xs, exs) in enumerate(zip(hist.data, hist.errors)):
        for y, (ys, eys) in enumerate(zip(xs, exs)):
            for z, (v, e) in enumerate(zip(ys, eys)):
                i = (x + 1, y + 1, z + 1)
                binval = h3f.GetBinContent(*i)
                binerr = h3f.GetBinError(*i)
                assert v == binval
                # assert np.isclose(e, np.sqrt(v))
                # assert np.isclose(e, binerr)

    h3f.Delete()

def test_histogram_sets_errors_1d(use_sumw2):
    from ROOT import TH1F
    h = TH1F("h3d", "TEST", 100, 0, 1)
    if use_sumw2:
        h.Sumw2()
    for x in np.random.random(10000):
        h.Fill(x)

    hist = Histogram.BuildFromRootHist(h)
    for i, (x, e) in enumerate(zip(hist.data, hist.errors)):
        assert x == h.GetBinContent(i + 1)
        assert e == h.GetBinError(i + 1)
        assert np.isclose(e, np.sqrt(x))


def test_histogram_sets_errors_2d(use_sumw2):
    from ROOT import TH2F
    h = TH2F("h3d", "TEST", 7, 0, 1, 3, 0, 1)
    if use_sumw2:
        h.Sumw2()
    for x, y in np.random.random((1000, 2)):
        h.Fill(x * 2, y)

    hist = Histogram.BuildFromRootHist(h)
    for i, j, (v, e) in enumerate_2d(hist.data, hist.errors):
        assert v == h.GetBinContent(i + 1, j + 1)
        assert e == h.GetBinError(i + 1, j + 1)
        assert np.isclose(e, np.sqrt(v))


def test_histogram_sets_errors_3d(use_sumw2):
    from ROOT import TH3F
    h = TH3F("h3d", "TEST", 7, 0, 1, 3, 0, 1, 5, 0, 1)
    if use_sumw2:
        h.Sumw2()
    for x, y, z in np.random.random((1000, 3)):
        h.Fill(x, y, z)

    hist = Histogram.BuildFromRootHist(h)
    for i, j, k, (v, e) in enumerate_3d(hist.data, hist.errors):
        assert v == h.GetBinContent(i + 1, j + 1, k + 1)
        assert e == h.GetBinError(i + 1, j + 1, k + 1)
        assert np.isclose(e, np.sqrt(v))


def enumerate_2d(*args):
    for i, a, in enumerate(zip(*args)):
        for j, b in enumerate(zip(*a)):
            yield i, j, b

def enumerate_3d(*args):
    for i, j, a in enumerate_2d(*args):
        for k, b in enumerate(zip(*a)):
            yield i, j, k, b


def test_histogram_constructor():
    np.random.seed(42)
    data = np.random.random(45)
    axis = np.linspace(0.0, 1.0, num=45)
    hist = Histogram.BuildFromData(data, axis)
    assert len(hist.axes) is 1

    # x_axis = hist.axes[0]
    # assert len(x_axis._bin_centers) is 45
    # assert x_axis[2] == 45/990

def test_1d_histogram_fill():
    # hist = TH1D()
    pass

@pytest.fixture
def ph1():
    min, max = 0, 20
    hist = Histogram(np.array(10, min, max))
    hist.fill(np.random.poisson(5, 10000))
    return hist


def test_histogram_addition():
    np.random.seed(42)
    a = np.random.random(45)
    b = np.random.random(45)
    axis = np.linspace(0.0, 1.0, num=45)
    hist_a = Histogram.BuildFromData(a, axis)
    hist_b = Histogram.BuildFromData(b, axis)
    hist_c = hist_a + hist_b

    for ay, by, cy in zip(hist_a.data, hist_b.data, hist_c.data):
        assert cy == ay + by


def test_histogram_subtraction():
    np.random.seed(42)
    a = np.random.random(45)
    b = np.random.random(45)
    axis = np.linspace(0.0, 1.0, num=45)
    hist_a = Histogram.BuildFromData(a, axis)
    hist_b = Histogram.BuildFromData(b, axis)
    hist_c = hist_a - hist_b

    for ay, by, cy in zip(hist_a.data, hist_b.data, hist_c.data):
        assert cy == ay - by


def test_histogram_subtraction_again():
    np.random.seed(42)

    hist_a = Histogram(45, 0, 1.0)
    hist_b = Histogram(45, 0, 1.0)

    for x in np.random.random(450):
        hist_a.fill(x)

    for x in np.random.random(500):
        hist_b.fill(x)

    hist_c = hist_a - hist_b

    for ay, by, cy in zip(hist_a.data, hist_b.data, hist_c.data):
        assert cy == ay - by

def test_histogram_division():
    np.random.seed(42)
    a = np.random.random(45)
    b = np.random.random(45)
    axis = np.linspace(0.0, 1.0, num=45)
    hist_a = Histogram.BuildFromData(a, axis)
    hist_b = Histogram.BuildFromData(b, axis)
    hist_c = hist_a / hist_b

    for ay, by, cy in zip(hist_a.data, hist_b.data, hist_c.data):
        assert cy == ay / by


def test_histogram_division_by_float():
    np.random.seed(42)
    hist_a = Histogram(45, 0.0, 1.0)
    hist_b = Histogram(45, 0.0, 1.0)
    scale = 5.5

    for x in np.random.random(450):
        hist_a.fill(x)
        hist_b.fillw(x, 1 / scale)
    hist_c = hist_a / scale

    for ay, by, cy in zip(hist_a.data, hist_b.data, hist_c.data):
        assert cy == ay / 5.5
        assert cy - by < 1e-15
