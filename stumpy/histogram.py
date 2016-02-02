#
# stumpy/histogram.py
#
"""
Histogram helper classes/methods.
"""

import numpy as np

import re
import functools
import itertools as itt
import ROOT
from stumpy.utils import ROOT_TO_NUMPY_DTYPE


class Histogram:
    """
    A class imitating the TH[1-3]D root histograms using numpy data structures.
    The class is designed to be built from and serialized to ROOT data
    structures.
    """

    def __init__(self, data, axis, errors=None, name=None, **kwargs):
        """
        Construct the histogram from data.

        Parameters
        ----------
        data : np.ndarray
            The data contained in the histogram.
        axes : np.array or collection of np.array
            Arrays of data describing the axes of the histogram. The number of
            axes must match the dimension of data, and the length of the axis
            must match the number of bins in the data.
        errors : np.ndarray, optional
            If specified, the error array holds the error in each bin of the
            histogram. If shape does not match data, a ValueError is raised.
            If no error parameter is given, the errors are assumed to be the
            square root of the data values (Poisson distribution).
        name : str
            Name of the histogram, used for serialization, corresponding to
            the ROOT name.

        Keyword Arguments
        -----------------
        overflow : np.ndarray
            Pairs of arrays describing the underflow/overflow bins of the
            histogram.

        Raises
        ------
        ValueError
            The shape of the data and error arrays do not match.
        """
        if np.ndim(data) == 1:
            self.axes = (Histogram.Axis(axis),)
        else:
            self.axes = tuple(Histogram.Axis(axis_data) for axis_data in axis)
        self.name = name

    @classmethod
    def BuildFromData(cls, data, errors=None):
        self = cls()
        self.data = root_numpy.hist2array(self._ptr, include_overflow=True)
        # add two overflow bins
        # error_shape = np.array(list(self.data.shape)) + [2, 2, 2]
        error_shape = self.data.shape
        errors = root_numpy.array(self._ptr.GetSumw2())
        self.error = np.sqrt(errors).reshape(error_shape)
        # print(">>", self.error[4, 4, 4])
        # print(">>", hist.GetBinError(4, 4, 4))
        self._axes = Histogram.Axis.BuildFromHist(hist)
        self._axis_data = np.array(list(
            [axis.GetBinCenter(i) for i in range(1, axis.GetNbins() + 1)]
            for axis in self._axes
        ))
        assert self.data.shape == tuple(a.data.shape[0] for a in self._axes)
        self.mask = Histogram.Mask(self)
        return self


    @classmethod
    def BuildFromRootHist(cls, hist):
        # typecheck
        if not isinstance(hist, ROOT.TH1):
            raise TypeError("Not a root histogram")

        for next_class in hist.__class__.__mro__:
            classname = next_class.__name__
            m = re.search('TH[1-3](?P<root_type>[CSIFD])', classname)
            if m:
                dtype = ROOT_TO_NUMPY_DTYPE[m.group('root_type')]
                break
        else:
            raise TypeError("Not a root histogram")

        if isinstance(hist, ROOT.TH3):
            nbins = (hist.GetNbinsZ() + 2, hist.GetNbinsY() + 2, hist.GetNbinsX() + 2)
            data = np.ndarray(shape=nbins, dtype=dtype, buffer=hist.GetArray()).T
            underflow = data[0, :, :], data[:, 0, :], data[:, :, 0]
            overflow = data[-1, :, :], data[:, -1, :], data[:, :, -1]
            data = np.copy(data[1:-1, 1:-1, 1:-1])
        elif isinstance(hist, ROOT.TH2):
            nbins = (hist.GetNbinsY() + 2, hist.GetNbinsX() + 2)
            data = np.ndarray(shape=nbins, dtype=dtype, buffer=hist.GetArray()).T
            underflow = data[0, :], data[:, 0]
            overflow = data[-1, :], data[:, -1]
            data = np.copy(data[1:-1, 1:-1])
        elif isinstance(hist, ROOT.TH1):
            nbins = hist.GetNbinsX()
            underflow = hist.GetBinContent(0)
            overflow = hist.GetBinContent(nbins + 1)
            buffer = hist.GetArray()
            buff_array = np.frombuffer(buffer, dtype=dtype, count=nbins+2)
            data = np.copy(buff_array[1:-1])

        self = cls.__new__(cls)
        self.data = data
        self.name = hist.GetName()
        self.title = hist.GetTitle()
        self.overflow = overflow
        self.underflow = underflow

        return self

        self._ptr = hist
        self.data = root_numpy.hist2array(self._ptr, include_overflow=True)
        # add two overflow bins
        # error_shape = np.array(list(self.data.shape)) + [2, 2, 2]
        error_shape = self.data.shape
        errors = root_numpy.array(self._ptr.GetSumw2())
        self.error = np.sqrt(errors).reshape(error_shape)
        # print(">>", self.error[4, 4, 4])
        # print(">>", hist.GetBinError(4, 4, 4))
        self._axes = (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis())
        self._axes = Histogram.Axis.BuildFromHist(hist)
        self._axis_data = np.array(list(
            [axis.GetBinCenter(i) for i in range(1, axis.GetNbins() + 1)]
            for axis in self._axes
        ))
        assert self.data.shape == tuple(a.data.shape[0] for a in self._axes)
        self.mask = Histogram.Mask(self)
        return self

    @property
    def shape(self):
        return self.data.shape

    # @functools.lru_cache
    def __getitem__(self, val):
        """
        Returns the value of bin specified by value val. If val is a float, the
        corresponding bin is searched for automatically and the value returned
        """
        if isinstance(val, tuple):
            val = tuple(axis.getbin(v) for v, axis in zip(val, self._axes))
        else:
            val = self._axes[0].getbin(val)
        return self.data[val]

    def find_bin(self, val):
        """

        Return the bin number, or tuple, which contains the value val. If the
        bin falls  outside the range of this histogram, this will return either
        a  decimal.Underflow or decimal.Overflow instance.

        Parameters
        ----------
        val : float or tuple of floats
            The value to look for in each axis.
        """
        def searchsorted(self, value, side, sorter):
            return self.data.searchsorted(value, side, sorter)

    def domain(self, *ranges):
        if ranges is ():
            domains = iter(axis.domain() for axis in self._axes)
        else:
            axis_range_pairs = zip(self._axes, ranges)
            domains = iter(axis.domain(r) for axis, r in axis_range_pairs)
        return itt.product(*domains)

    def bounded_domain(self, *ranges):
        """
        Return a numpy array containing the bin centers of the range that
        this axis' domain.
        """
        if ranges is ():
            domains = iter(axis.domain() for axis in self._axes)
        else:
            axis_range_pairs = zip(self._axes, ranges)
            domains = iter(axis.bounded_domain(r) for axis, r in axis_range_pairs)

        return np.array([l for l in itt.product(*domains)])

    def bin_ranges(self, *ranges):
        """
        zips ranges with axes to generate bin_ranges
        """
        return tuple(a.getbin(r) for a, r in zip(self._axes, ranges))

    def centered_bin_ranges(self, *ranges, expand=False, inclusive=False):
        """
        Applies centered_bin_range_pair to each axis. Returns tuple of integer
        pairs, or if expand is True, returns one flattened tuple of ints.
        """
        res = []
        for r, a in zip(ranges, self._axes):
            if r is not None:
                val = a.centered_bin_range_pair(*r)
                if inclusive:
                    val = val[0], val[1] - 1
                res.append(val)
            else:
                res.append(((), ()))

        if expand:
            return tuple(x for x in res for x in x)
        return res

    def bin_at(self, x, y=0.0, z=0.0):
        return tuple(axis.bin_at(a) for axis, a in zip(self._axes, (x, y, z)))

    def getslice(self, x, y=0.0, z=0.0):
        return tuple(axis.getslice(a) for axis, a in zip(self._axes, (x, y, z)))

    def value_at(self, x, y=0.0, z=0.0):
        i, j, k = self.bin_at(x, y, z)
        return self.data[i, j, k]

    def value_in(self, i, j=0, k=0):
        return self.data[i, j, k]

    def project_1d(self, axis_idx, *axis_ranges, bounds=(None, None)):
        """
        Project multi-dimensional data into one dimention along axis with
        index 'axis_idx'. The variable 'axis_ranges' parameter limits the
        range of all other axes, with the position of each axis_range
        corresponding to each axis NOT the axis being projected into.
        For example:

            # projects onto x-axis, y is limited between (1.0, 2.0), z (-1.0, 1.0)
            hist.project_1d(0, (1.0, 2.0), (-1.0, 1.0))

            # projects onto y-axis, x is limited between (1.0, 2.0), z (-1.0, 1.0)
            hist.project_1d(1, (1.0, 2.0), (-1.0, 1.0))

        The optional 'bounds' variable is the limit of the projected axis; this
        defaults to no-limit
        """
        assert 0 <= axis_idx < self.data.ndim

        # merge specified boundries with infinite slice(None) generator
        bounds = itt.chain(axis_ranges, itt.repeat(slice(None)))

        ranges = []
        summed_axes = []

        axes = self._axes
        for i, axis in enumerate(self._axes):
            if i == axis_idx:
                ranges.append(axis.getslice(bounds))
            else:
                s = axis.getslice(next(bounds))
                ranges.append(s)
                if isinstance(s, slice):
                    summed_axes.append(i)

        res = self.data[ranges].sum(axis=tuple(summed_axes))
        return res

    def project_2d(self, axis_x, axis_y, *axis_ranges, bounds_x=slice(None), bounds_y=slice(None)):
        """
        Project the histogram into 2 dimensions.
        """
        assert axis_x != axis_y
        assert 0 <= axis_x < self.data.ndim
        assert 0 <= axis_y < self.data.ndim

        bounds = itt.chain(axis_ranges, itt.repeat(slice(None)))
        ranges = []
        summed_axes = []
        for i, axis in enumerate(self._axes):
            if i == axis_x:
                ranges.append(axis.getslice(bounds_x))
            elif i == axis_y:
                ranges.append(axis.getslice(bounds_y))
            else:
                s = axis.getslice(next(bounds))
                ranges.append(s)
                if isinstance(s, slice):
                    summed_axes.append(i)

        return self.data[ranges].sum(axis=tuple(summed_axes))

    def __str__(self):
        return '<{dim}D Histogram "{name}" ({sizes}) at {id}>'.format(
            name=self._ptr.GetName(),
            dim=self.data.ndim,
            sizes="-".join(map(str, self.data.shape)),
            id="0x%x" % id(self),
        )

    #
    # Math Functions
    #
    def __truediv__(self, rhs):
        if isinstance(rhs, Histogram):
            quotient = self._ptr.Clone()
            quotient.Divide(rhs._ptr)
            q = Histogram.BuildFromRootHist(quotient)
            return q
        elif isinstance(rhs, float):
            clone = self._ptr.Clone()
            clone.Scale(1.0 / rhs)
            return Histogram.BuildFromRootHist(clone)
        else:
            raise TypeError("Cannot divide histogram by %r" % rhs)

    class Axis:
        """
        Histogram axis class. Contains all information about binning, including
        labels. Imitates the ROOT class TAxis.
        """

        def __init__(self, data, ):
            self._bin_centers = data  # np.array(data)
            return
            if not self._ptr.IsVariableBinSize():
                maxbin = self._ptr.GetNbins() + 1
                self.data = np.linspace(self._ptr.GetBinCenter(0),
                                        self._ptr.GetBinCenter(maxbin),
                                        maxbin + 1)
                assert all(self._ptr.GetBinCenter(i) - self.data[i] < 1e-9
                           for i in (0, 1, maxbin // 2, maxbin - 2))
            else:
                self.data = root_numpy.array(self._ptr.GetXbins())

        def searchsorted(self, value, side, sorter):
            return self.data.searchsorted(value, side, sorter)

        def search(self, value):
            idx = np.searchsorted(self.data, value, side="left")
            rval, lval = array[idx - 1:idx + 1]
            return rval if fabs(value - rval) < fabs(value - lval) else lval

        @classmethod
        def BuildFromHist(self, hist):
            """
            Returns tuple of 3 Axis objects, corresponding to the x,y,z axes of
            the hist argument
            """
            axes = (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis())
            return tuple(map(Histogram.Axis, axes))

        # def __getattr__(self, attr):
        #     """
        #     Forwards any attribute requests to the real axis object
        #     """
        #     return getattr(self._ptr, attr)

        def __getitem__(self, index):
            """
            Returns the value of bin specified by index
            """
            return self._bin_centers[index]

        def bin_at(self, value):
            self._ptr
            return self._ptr.FindBin(value)

        def getbin(self, value):
            """
            Return the bin relating to value
            """
            if isinstance(value, float):
                return self._ptr.FindBin(value)
            if isinstance(value, slice):
                return slice(*map(self.getbin, (value.start, value.stop)))
            if isinstance(value, (tuple, list)):
                start, stop = map(self.getbin, value)
                if isinstance(stop, int):
                    stop += 1
                return slice(start, stop)

            return value

        def getslice(self, value):
            """
            Alias of getbin
            """
            return self.getbin(value)

        def bin_range_pair(self, value):
            """
            Returns a pair (tuple) of integers representing the beginning
            and (exclusive) ending of the range of bins containing the value(s)
            in value. If value is a floating point, this returns the bin in
            which this value can be found, and the 'next' bin.
            """
            asbins = self.getbin(value)
            if isinstance(asbins, int):
                start = asbins
                stop = asbins + 1
            elif isinstance(asbins, slice):
                start = asbins.start
                stop = asbins.start + 1
            else:
                raise ValueError("Cannot find bin_range_pair of %s" % (value))
            return start, stop

        def bin_range(self, value):
            """
            Returns a range object which returns the bins specified by the
            argument. If value is a floating point number, the range will return
            only the bin the value falls into. If value is an integer, range
            will only return that value. If value is a pair (tuple or list) of
            floats or ints, the range interates through all bins falling
            between the two bins.
            """
            start, stop = self.bin_range_pair(value)
            return range(start, stop)

        def centered_bin_range_pair(self, value=0.0, width=1):
            """
            Returns a pair (tuple) of integers representing the beginning
            and (exclusive) ending of the range of bins centered around the
            bin containing the value parameter. The width parameter shifts the
            bins above and below the value.

            The range will be $2 * 'width' + 1$ long.
            """
            start, stop = self.bin_range_pair(value)
            start -= width
            stop += width
            return start, stop

        def domain(self, n=None):
            """
            Return a numpy array containing the floating point values within
            this axis' domain. If no 'n' parameter is specified, this is a copy
            of the data axis' data
            """
            if n is None:
                return np.copy(self.data)
            else:
                return np.linspace(n, self.data[0], self.data[-1])

        def bounded_domain(self, value):
            """
            Return a numpy array containing the bin centers of the range that
            this axis' domain.
            """
            s = self.getslice(value)
            return self.data[s]


    class Mask:
        """
        A class wrapping a numpy array-mask used for keeping same shape between
        data and error arrays
        """

        def __init__(self, hist):
            self.hist = hist
