#
# stumpy/histogram.py
#
"""
Histogram helper classes/methods.
"""

import numpy as np

import re
import functools
from decimal import Underflow, Overflow
import itertools as itt
from itertools import zip_longest
from copy import copy
from stumpy.utils import ROOT_TO_NUMPY_DTYPE


class Histogram:
    """
    A class imitating the TH[1-3]D root histograms using numpy data structures.
    The class is designed to be built from and serialized to ROOT data
    structures.
    """

    DEFAULT_AXIS_DATATYPE = 'low_edge'

    def __init__(self, *axis_parameters, name=None, **kwargs):
        """
        Construct a histogram.

        The variable

        Parameters
        ----------
        axis_parameters : tuple of axis parameters
            Used to create axes of the histogram. The histogram's dimension is
            interpreted from this parameter.

            If the first value is an interger, the parameter is expected to be
            a tuple of 3 numbers (nbins, min, max) which will be used to
            construct the x-axis. This is only used for the 1D case.
            Example: ``hist = Histogram(10, 0.0, 1.0)``

            All higher dimensions must group axis parameters; the number of
            elements in the axis_parameters tuple is the dimension of the
            histogram. Any object that may be used to construct an Axis may be
            used as an argument. If the object is an instance of Axis, this
            axis is copied.
            Example: ``hist = Histogram((20, 0.0, 1.0), (10, 0.0, 1.0))``

        Keyword Arguments
        -----------------
        name : str
            Identifier used for serialization
        x_params : dict
            Keyword arguments passed to the first Axis constructor.
        axis_params : tuple of dict
            Keyword arguments passed to the matching Axis constructor.

        Examples
        --------
        .. code:: python
            # Create 1D histogram with 100 bins from 0 to 10
            hist = Histogram(100, 0.0, 10.0)

        .. code:: python
            # Create 2D histogram,  xaxis: 50 bins from 0 to 10
            #                       yaxis: 20 bins from 0.3 to 0.8
            hist_2d = Histogram((50, 0.0, 10.0), (20, 0.3, 0.8))

        """
        if len(axis_parameters) == 0:
            raise ValueError("Histogram must be given axis_parameters")
        elif isinstance(axis_parameters[0], int):
            newaxis = Histogram.Axis.BuildWithLinearSpacing
            axis_kwargs = kwargs.get('x_params', {})
            axis_kwargs.update(kwargs.get('axis_params', ({}, ))[0])
            self.axes = (newaxis(*axis_parameters, **axis_kwargs), )
        else:
            self.axes = tuple(Histogram.Axis(axis_param)
                              for axis_param in axis_parameters)

        shape = tuple(axis.nbins for axis in self.axes)
        self.data = np.zeros(shape)
        self._errors = None
        self.underflow, self.overflow = 0.0, 0.0

    @classmethod
    def BuildFromData(cls, data, axis, errors=None, name=None, **kwargs):
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
        self = cls.__new__(cls)
        if isinstance(axis, tuple) and len(axis) == 1:
            self.axes = (Histogram.Axis(axis[0]), )
        elif np.ndim(data) == 1:
            self.axes = (Histogram.Axis(axis), )
        else:
            self.axes = tuple(Histogram.Axis(axis_data) for axis_data in axis)
        self.name = name
        self.data = data
        self.errors = errors if (errors is not None) else np.sqrt(self.data)
        return self

    @classmethod
    def BuildFromRootHist(cls, hist):
        import ROOT
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
    def errors(self):
        if self._errors is None:
            return np.sqrt(self.data)
        return self._errors

    @errors.setter
    def errors(self, errs):
        assert errs.shape == self.data.shape
        self._errors = errs

    @property
    def shape(self):
        return self.data.shape

    def fill(self, *data):
        """
        Fill bin found at 'data' by an unweighted value of 1.0.
        """
        try:
            bins = tuple(a.getbin(d) for a, d in zip_longest(self.axes, data))
        except:
            raise
        try:
            self.data[bins] += 1.0
        except IndexError:
            if bins[0] == Underflow:
                self.underflow += 1.0
            elif bins[0] == Overflow:
                self.overflow += 1.0

        return bins

    def fill_all(self, data):
        """
        Fill histogram with all values in data. This is a helper method to
        automatically fill the histogram with a collection of data.

        For example, if this is a two dimensional histogram, create a numpy
        array with shape (X, 2) to fill the histogram with X pairs.

        Parameters
        ----------
        data : iterable
            Data is an iterable returning N numbers, where N is dimension of
            histogram.
        """
        for d in data:
            bins = tuple(a.getbin(d) for a, d in zip_longest(self.axes, data))
            self.data[bins] += 1.0

    def fillw(self, *data, weight=None):
        """
        Fill histogram bin containing 'data' with weight.

        There are two ways to use this method - the weight keyword argument is
        None, and data is an array of N + 1 dimensions, where N is dimension of
        this histogram, where the last value in data is the weight

        Or weight is NOT none, and data is an N dimension collection.


        """
        if weight is None:
            *data, weight = data
        bins = tuple(a.getbin(d) for a, d in zip_longest(self.axes, data))
        try:
            self.data[bins] += weight
        except IndexError:
            if bins[0] == Underflow:
                self.underflow += weight
            elif bins[0] == Overflow:
                self.overflow += weight
        return bins

    def __copy__(self):
        """
        Create a copy of this histogram. Essentially calls np.copy on all data
        structure.
        """
        the_copy = Histogram.BuildFromData(data=np.copy(self.data),
                                           errors=np.copy(self.data),
                                           axis=self.axes)
        return the_copy

    def __getitem__(self, val):
        """
        Returns the value of bin containing the value val. If val is an
        integer, this is interpreted as a bin number, and the value at index
        val is returned.  If val is a float, the corresponding bin is searched
        for automatically and the value returned. If the val is tuple of values


        Examples
        --------
        # get value at 4.2
        hist[4.2]

        # get value in two dimensional hist
        hist2d[4.2, 5.0]

        # get numpy array of values between 0.5 and 1.0 - inclusive
        hist2d[0.5:1.0]
        """
        if isinstance(val, int):
            return self.data[val]
        elif isinstance(val, tuple):
            bins = tuple(axis.getbin(v) for v, axis in zip(val, self._axes))
            return self.data[bins]
        else:
            return self.data[self.axis.getbin(val)]

    def AsRootHist(self, **kwargs):
        """
        Return a ROOT histogram which is equivalent of this.
        """
        import ROOT
        hist_classname = 'TH%d%c' % (self.data.ndim, {
                                        np.int64: 'I',
                                        np.int32: 'I',
                                        np.float32: 'F',
                                        np.float64: 'D',
                                     }[self.dtype])
        hist_class = getattr(ROOT, hist_classname)
        axis_info = np.array([(axis.nbins, axis.min, axis.max)
            for axis in self.axes
        ]).flatten()

        hist = hist_class(self.name, self.title, *axis_info)
        return hist

    def find_bin(self, val):
        """

        Return the bin number, or tuple, which contains the value val. If the
        bin falls  outside the range of this histogram, this will return either
        a Underflow or Overflow instance.

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

    def bin_at(self, *x):
        """
        Find and return the bin location which contains the value 'x'. The
        number of values in x must equal the dimension of the histogram.
        """
        return tuple(axis.bin_at(a) for axis, a in zip_longest(self._axes, x))

    def getslice(self, *x):
        """
        Find and return the bin location which contains the value 'x'. The
        number of values in x must equal the dimension of the histogram.
        """
        get_slice = Histogram.Axis.getslice
        return tuple(itt.starmap(get_slice, zip_longest(self._axes, x)))

    def value_at(self, x, y=0.0, z=0.0):
        """
        Return the value stored in the bin which contains the value 'x'. The
        number of values in x must equal the dimension of the histogram.
        """
        a_bin = self.bin_at(x, y, z)
        return self.data[a_bin]

    def value_in(self, *i):
        """
        Return the value located in bin 'i'. This is equivalent to using the
        [] operator with a tuple of integers.
        """
        return self.data[i]

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
    def __radd__(self, lhs):
        """
        Rightside add. Applies standard (lefthand) addition, as addition is a
        communitive operation on histograms.
        """
        return self.__add__(lhs)

    def __add__(self, rhs):
        """
        Create a new histogram which is the result of the addition of the two
        histograms. The histograms must have the same shape and bins.

        Errors are propagated as expected:
        .. math::
            error =

        Raises
        ------
        ValueError
            If the two histograms have different shape.
        """
        hist = copy(self)
        hist += rhs
        return hist

    def __iadd__(self, rhs):
        """
        In place add method.
        """
        self.data += rhs.data
        self.errors = self.errors ** 2 + rhs.errors  ** 2
        return self

    def add(self, rhs, scale=1.0):
        """
        Explicit add method, able to automatically scale the right hand side.
        """
        if scale == 1.0:
            return self + rhs

        hist = rhs / scale
        hist += self
        return hist

    def __sub__(self, rhs):
        """
        Histogram Subtraction
        """
        hist = copy(self)
        hist -= rhs
        return hist

    def __isub__(self, rhs):
        """
        Inplace Histogram Subtraction
        """
        self.data -= rhs.data
        self.errors = self.errors ** 2 + rhs.errors  ** 2
        return self

    def __truediv__(self, rhs):
        """
        Divide histogram.
        If right hand side is a number, this simply scales the histogram. If
        right hand side is another histogram, this will do bin-by-bin division,
        adding errors appropriately.
        """
        if isinstance(rhs, Histogram):
            quotient = copy(self)
            quotient /= rhs
            return quotient
        elif isinstance(rhs, float):
            quotient = copy(self)
            # quotient.Scale(1.0 / rhs)
            quotient.data /= rhs
            return quotient
        else:
            raise TypeError("Cannot divide histogram by %r" % rhs)

    def __itruediv__(self, rhs):
        """
        Inplace histogram division
        """
        if isinstance(rhs, Histogram):
            self.data /= rhs.data
            self.errors = self.errors * self.data + rhs.errors * rhs.data / (rhs.data + self.data)
        elif isinstance(rhs, float):
            self.data /= rhs
            self.errors /= rhs
        return self

    class Axis:
        """
        Histogram axis class. Contains all information about binning, including
        labels. Imitates the ROOT class TAxis.

        Unlike the ROOT class, this class provides a method for determining
        whether the associated values of each bin should be interpreted to be
        the low edge, high edge, or center.


        """

        def __init__(self, data=None, **kwargs):
            """
            Keyword Args
            ------------
            labels : list of str
                List of strings, offering ability to add text labels to bins
            low_edges : np.array
                Array of 'low' edges - conflicts with high_edges, centers
            centers : np.array
                Array of 'center' values - conflicts with low_edges, high_edges
            high_edges : np.array
                Array of 'high' edges - conflicts with low_edges, centers

            count_min_max : tuple
                Tuple containing the number of bins, the low x value, and the
                high x value.
            """
            if sum(key in ('low_edges', 'centers', 'high_edges')
                   for key in kwargs.keys()) > 1:
                raise ValueError("Axis can only be set from ONE of low_edges, "
                                 "centers, and high_edges")
            if 'low_edges' in kwargs:
                self._low_edges = np.copy(kwargs['low_edges'])
                self._low_edges.flags.writeable = False
                self._bin_width = self._low_edges[1] - self._low_edges[0]
            elif 'high_edges' in kwargs:
                self._high_edges = np.copy(kwargs['high_edges'])
                self._high_edges.flags.writeable = False
                self._bin_width = self._high_edges[1] - self._high_edges[0]
                self._low_edges = self._high_edges - self._bin_width
            elif 'centers' in kwargs:
                self._centers = np.copy(kwargs['centers'])
                self._centers.flags.writeable = False
                self._bin_width = self._low_edges[1] - self._low_edges[0]
                self._low_edges = self._centers - self._bin_width / 2.0
            elif 'count_min_max' in kwargs:
                nbins, self._xmin, self._xmax = kwargs['count_min_max']
                self._low_edges = np.linspace(self._xmin, self._xmax, nbins, endpoint=False)
                self._bin_width = self._low_edges[1] - self._low_edges[0]
                self._low_edges.flags.writeable = False
            elif isinstance(data, Histogram.Axis):
                self._low_edges = data._low_edges
                self._bin_width = self._low_edges[1] - self._low_edges[0]
            else:
                print("DAA:", data)
                self._low_edges = data
                self._bin_width = self._low_edges[1] - self._low_edges[0]
            self._xmin = self._low_edges[0]
            self._xmax = self._low_edges[-1] + self._bin_width

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
        def BuildWithLinearSpacing(cls, nbins, min_x, max_x, **kwargs):
            self = cls(count_min_max=(nbins, min_x, max_x,), **kwargs)
            return self

        @classmethod
        def BuildFromROOTAxis(cls, axis):
            nbins = axis.GetNbins()
            if axis.IsVariableBinSize():
                bin_array = np.frombuffer(axis.GetXbins(), dtype='f8', count=nbins)
                self = cls(bin_array)
            else:
                self = cls.BuildWithLinearSpacing(nbins,
                                                  axis.GetXmin(),
                                                  axis.GetXmax())
                bin_array = ()

        @classmethod
        def BuildAxisTupleFromRootHist(cls, hist):
            """
            Returns tuple of 3 Axis objects, corresponding to the x,y,z axes of
            the hist argument
            """
            axes = (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis())
            return tuple(map(cls.BuildFromROOTAxis, axes))

        @property
        def nbins(self):
            return len(self._low_edges)

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
                idx = np.searchsorted(self._low_edges, value)
                if idx == 0:
                    return Underflow
                elif idx >= len(self._low_edges):
                    return Overflow
                else:
                    return idx
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
