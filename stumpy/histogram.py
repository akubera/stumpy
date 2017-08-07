#
# stumpy/histogram.py
#
"""
Histogram helper classes/methods.
"""

import numpy as np

import uuid
import operator as op
import itertools as itt

from copy import copy
from operator import mul
from functools import reduce
from itertools import zip_longest
from .utils import root_histogram_datatype, get_root_object, enumerate_histogram

from .axis import Axis, MultiAxis, Overflow, Underflow


def root_histogram_shape(root_hist, use_matrix_indexing=True):
    """
    Return a tuple corresponding to the shape of the histogram.
    If use_matrix_indexing is true, the tuple is in 'reversed' zyx
    order. Matrix-order is the layout used in the internal buffer
    of the root histogram - keep True if reshaping the array).
    """
    dim = root_hist.GetDimension()
    shape = np.array([root_hist.GetNbinsZ(),
                      root_hist.GetNbinsY(),
                      root_hist.GetNbinsX()][3-dim:]) + 2
    if not use_matrix_indexing:
        shape = reversed(shape)
    return tuple(shape)


class Histogram:
    """
    A class imitating the TH[1-3]D root histograms using numpy data structures.
    The class is designed to be built from and serialized to ROOT data
    structures.
    """

    # not used
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
        self.axes = MultiAxis.FromData(axis_parameters)
        self.data = np.zeros(self.axes.shape)
        self._errors = None
        self.underflow, self.overflow = 0.0, 0.0
        self.title = kwargs.pop('title', '<Histogram Title>')
        self.name = kwargs.pop('name', None)

    @classmethod
    def BuildFromData(cls, data, axes, errors=None, name=None, **kwargs):
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
        title : str
            Title to be displayed above the plot

        Raises
        ------
        ValueError
            The shape of the data and error arrays do not match.
        """
        self = cls.__new__(cls)
        self.axes = copy(axes) if isinstance(axes, MultiAxis) else\
                    MultiAxis.FromData(axes)
        assert self.axes.shape == np.shape(data), "Data and axes shape mismatch "\
            "(%s ≠ %s)" % (self.axes.shape, np.shape(data))
        self.name = name
        self.data = np.array(data)
        self.underflow = 0.0
        self.overflow = 0.0
        self.title = kwargs.pop('title', '')
        self._errors = errors
        return self

    @staticmethod
    def hist_data_to_array(root_hist, use_matrix_indexing=False):
        """
        Returns an array containing all data in the histogram.
        Note: this uses coordinate indexing: hist[x, y, z] instead of
        *matrix* indexing, [z, y, x], which may be expected.
        A transpose will switch from one to the other.
        Note: This returns ALL data stored in ROOT data buffer,
        including overflow, not including errors.

        Parameters
        ----------
        root_hist : (TH1|TH2|TH3)
            ROOT histogram type of 1-3 dimension
        use_matrix_indexing : bool
            If true, data is returned in matrix-indexed form, where
            the most significant index (i.e. the leftmost) is 'z'; otherwise returns
            histogram in coordinate form, where most significant
            index is 'x'.

        """
        data = np.ndarray(shape=root_histogram_shape(root_hist),
                          dtype=root_histogram_datatype(root_hist),
                          buffer=root_hist.GetArray())

        # transpose data to make coordinate-based-indexing
        if not use_matrix_indexing:
            data = data.T
        return data

    @staticmethod
    def hist_errors_to_array(root_hist, use_matrix_indexing=False):
        """
        Return the errors, following same convensions as
        `hist_data_to_array`.
        """
        dtype = np.double
        shape = root_histogram_shape(root_hist)
        err_buffer = root_hist.GetSumw2()

        # If histogram has precalculated errors, just copy the buffer
        if err_buffer.GetSize() != 0:
            sumw2 = np.ndarray(shape=shape,
                               dtype=dtype,
                               buffer=err_buffer.GetArray())
            errs = np.sqrt(np.copy(sumw2))
        else:
            err_it = map(root_hist.GetBinError, range(reduce(mul, shape, 1)))
            errs = np.fromiter(err_it, dtype=dtype).reshape(shape)

        # transpose data to make coordinate-based-indexing
        if not use_matrix_indexing:
            errs = errs.T
        return errs

    @staticmethod
    def split_underflow_and_overflow_from(data):
        """
        Splits the bins pertaining to overflow/underflow from some
        multi-dimentional array.

        The data is copied.
        """
        if data.ndim == 1:
            underflow = data[0]
            overflow = data[-1]
            data = np.copy(data[1:-1])
        elif data.ndim == 2:
            underflow = data[0, :], data[:, 0]
            overflow = data[-1, :], data[:, -1]
            data = np.copy(data[1:-1, 1:-1])
        elif data.ndim == 3:
            underflow = data[0, :, :], data[:, 0, :], data[:, :, 0]
            overflow = data[-1, :, :], data[:, -1, :], data[:, :, -1]
            data = np.copy(data[1:-1, 1:-1, 1:-1])
        else:
            raise RuntimeError("Unexpected dimension %d of histogram data" % data.ndim)
        return data, underflow, overflow

    @classmethod
    def BuildFromRootHist(cls, root_hist):
        """
        Construct a stumpy Histogram object out of a standard ROOT
        Histogram TH1, TH2, TH3 of any type.

        Parameters
        ----------
        root_hist : ROOT.(TH1|TH2|TH3)
            The ROOT histogram containing the data
        """

        # copy histogram data into numpy arrays
        root_data = cls.hist_data_to_array(root_hist)
        root_errors = cls.hist_errors_to_array(root_hist)

        # create new Histogram object
        self = cls.__new__(cls)

        # split bins from overflow/underflow
        self.data, self.underflow, self.overflow = \
            cls.split_underflow_and_overflow_from(root_data)

        self._errors, *_ = \
            cls.split_underflow_and_overflow_from(root_errors)

        self.name = root_hist.GetName()
        self.title = root_hist.GetTitle()

        self.axes = MultiAxis.FromRootHistogram(root_hist)

        assert self.data.shape == self._errors.shape, 'data/error shape mismatch '\
            '(%s ≠ %s)' % (self.data.shape, self._errors.shape)

        return self

    def apply_mask(self, *masks):
        """
        Return new histogram with mask applied to data/errors/axis
        """
        masked_hist = self.__new__(self.__class__)
        masked_hist.name = self.name
        masked_hist.title = self.title
        masked_hist.data = self.data[masks]
        try:
            masked_hist._errors = self._errors[masks]
        except AttributeError:
            masked_hist._errors = None

        masked_hist.axes = self.axes.masked_by(*masks)
        masked_hist.underflow = None
        masked_hist.overflow = None
        return masked_hist

    def apply_slice(self, *slices):
        """
        Return new histogram with sliced data/errors/axis
        """
        slices = self.axes.get_slice(*slices)
        masked_hist = self.__new__(self.__class__)
        masked_hist.name = self.name
        masked_hist.title = self.title
        masked_hist.data = self.data[slices]
        try:
            masked_hist._errors = self._errors[slices]
        except AttributeError:
            masked_hist._errors = None

        # we can used masked here because its a little more
        # efficient and we've already called get_slice
        masked_hist.axes = self.axes.masked_by(*slices)
        assert masked_hist.shape == masked_hist.axes.shape
        masked_hist.underflow = None
        masked_hist.overflow = None
        return masked_hist

    def as_matrix(self):
        """
        Returns the data in matrix form, so data is addressed in [y, x]
        or [j, i] mode instead of [x, y] mode.
        This ends up being a simple transpose.
        """
        return self.data.T

    @property
    def errors(self):
        if self._errors is None:
            return np.sqrt(self.data)
        return self._errors

    @errors.setter
    def errors(self, errs):
        if np.shape(errs) != self.data.shape:
            raise ValueError("Attempting to set error array of incorrect shape "
                "({} ≠ {})".format(np.shape(errs), self.data.shape))
        self._errors = errs

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

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
        Fill histogram with all values in data.
        This is a helper method to automatically fill the histogram
        with a collection of data.

        For example, if this is a two dimensional histogram, create
        a numpy array with shape (X, 2) to fill the histogram with
        X pairs.

        Parameters
        ----------
        data : iterable
            Data is an iterable returning N numbers, where N is
            dimension of histogram.

        Returns
        -------
        None
        """
        shape = np.shape(data)
        dim = np.ndim(data)

        # only a value - just send to fill
        if dim == 0:
            self.fill(data)
        # 1D 'list' of values
        elif dim == 1 and self.data.ndim == 1:
            pass
        elif dim == 1 and shape == self.data.ndim:
            self
        elif dim == 2 and shape[1] == self.data.ndim:
            pass
        elif shape[1:] != self.data.ndim:
            raise ValueError(
                "Array given to fill_all() does not have compatible "
                "shape to fill the histogram: {received} != {expected}".format(
                    received=shape[1:], expected=self.data.ndim))

        dest = np.zeros(self.data.shape)
        for d in data:
            bins = tuple(a.getbin(d) for a, d in zip_longest(self.axes, data))
            dest[bins] += 1.0
        self.data += dest
        if self._errors:
            pass

    def fillw(self, *data, weight=None):
        """
        Fill histogram bin containing 'data' with weight.

        There are two ways to use this method - the weight keyword
        argument is None, and data is an array of N + 1 dimensions,
        where N is dimension of this histogram, where the last value
        in data is the weight.

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
        Create a copy of this histogram. Essentially calls np.copy on
        all data structure.
        """
        the_copy = Histogram.BuildFromData(data=np.copy(self.data),
                                           errors=np.copy(self.errors),
                                           axes=self.axes)
        the_copy.title = self.title
        the_copy.underflow = self.underflow
        the_copy.overflow = self.overflow
        return the_copy

    def __getitem__(self, val):
        """
        Returns the value of bin containing the value val.
        If val is an integer, this is interpreted as a bin number,
        and the value at index val is returned.
        If val is a float, the corresponding bin is searched for
        automatically and the value returned.
        If the val is tuple of values


        Examples
        --------
        # get value at 4.2
        hist[4.2]

        # get value in two dimensional hist
        hist2d[4.2, 5.0]

        # get numpy array of values between 0.5 and 1.0 - inclusive
        hist2d[0.5:1.0]
        """
        is_mask = isinstance(val, np.ndarray) and val.dtype == np.bool
        if isinstance(val, int) or is_mask:
            return self.data[val]
        elif isinstance(val, tuple):
            ranges = self.bin_ranges(*val)
            return self.data[ranges]
        elif isinstance(val, slice):
            start = self.x_axis.getbin(val.start)
            stop = self.x_axis.getbin(val.stop)
            return self.data[start:stop]
        else:
            i = self.axes.get_bin(val)
            if i == Underflow:
                return self.underflow
            elif i == Overflow:
                return self.overflow
            else:
                return self.data[i]

    def AsRootHist(self, **kwargs):
        """
        Return a ROOT histogram which is equivalent of this.
        """
        import ROOT
        hist_classname = 'TH%d%c' % (self.data.ndim, {
                                        np.dtype(np.int64): 'I',
                                        np.dtype(np.int32): 'I',
                                        np.dtype(np.float32): 'F',
                                        np.dtype(np.float64): 'D',
                                     }[self.data.dtype])
        hist_class = getattr(ROOT, hist_classname)
        name = kwargs.pop('name', (self.name or ""))
        title = kwargs.pop('title', (self.title or "<UNSET TITLE>"))

        hist = hist_class(str(self.name), str(self.title), *self.axes.bin_info)

        if self._errors is None:
            for i, (d, ) in enumerate_histogram(self, start=1):
                hist.SetBinContent(*i, d)
        else:
            for i, (d, e) in enumerate_histogram(self, start=1, with_errors=1):
                hist.SetBinContent(*i, d)
                hist.SetBinError(*i, e)

        # TODO: Implement copying overflow data
        # if hist.ndim == 1:
        #     hist.SetBinContent(0, self.underflow or 0)
        #     hist.SetBinContent(hist.shape[0] + 1, self.overflow or 0)
        #     hist.SetBinError(0, ?)
        #     hist.SetBinError(0, ?)
        # elif hist.ndim == 2:
        #   xsize, ysize = hist.shape
        #   ???

        return hist

    def draw_nb_ROOT(self, **opts):
        """
        Creates and displays an ipython html element containing the
        JSROOT container of the drawing.
        """
        from IPython.display import HTML, Javascript, display
        div_uid = uuid.uuid1()
        display(HTML("<div id='{id}'></div>".format(id=div_uid)))
        display(Javascript("""require(["JSRoot"], function (ROOT) {
            console.log('Drawing %s');
            console.log("jsroot::", ROOT);
            });
        """ % div_uid))

    def domain(self, *ranges):
        if ranges is ():
            domains = iter(axis.domain() for axis in self.axes)
        else:
            axis_range_pairs = zip(self.axes, ranges)
            domains = iter(axis.domain(r) for axis, r in axis_range_pairs)
        return itt.product(*domains)

    def bounded_domain(self, *ranges):
        """
        Return a numpy array containing the bin centers of the range
        that this axis' domain.
        """
        if ranges is ():
            domains = iter(axis.domain() for axis in self.axes)
        else:
            axis_range_pairs = zip(self.axes, ranges)
            domains = iter(axis.bounded_domain(r) for axis, r in axis_range_pairs)

        return np.fromiter(itt.product(*domains), np.float32)

    def bin_ranges(self, *ranges):
        """
        zips ranges with axes to generate bin_ranges
        """
        return tuple(a.getbin(r) for a, r in zip(self.axes, ranges))

    def centered_bin_ranges(self, *ranges, expand=False, inclusive=False):
        """
        Applies centered_bin_range_pair to each axis.
        Returns tuple of integer pairs, or if expand is True, returns
        one flattened tuple of ints.
        """
        res = []
        for r, a in zip(ranges, self.axes):
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
        Find and return the bin location which contains the value 'x'.
        The number of values in x must equal the dimension of the
        histogram.
        """
        if len(self.axes) == 1:
            return self.axes[0].bin_at(*x)
        return tuple(axis.bin_at(a) for axis, a in zip_longest(self.axes, x))

    def get_slice(self, *x):
        """
        Find and return the bin location which contains the value 'x'.
        The number of values in x must equal the dimension of the histogram.
        """
        get_slice = Axis.get_slice
        return tuple(itt.starmap(get_slice, zip_longest(self.axes, x)))

    def value_at(self, x, y=0.0, z=0.0):
        """
        Return the value stored in the bin which contains the value 'x'.
        The number of values in x must equal the dimension of the histogram.
        """
        a_bin = self.bin_at(x, y, z)
        return self.data[a_bin]

    def value_in(self, *i):
        """
        Return the value located in bin 'i'.
        This is equivalent to using the [] operator with a tuple of
        integers.
        """
        return self.data[i]

    def copy_data_with_overflow(self):
        result = np.hstack([[self.underflow],
                             self.data,
                             [self.overflow],
                            ])
        return np.ascontiguousarray(result)

    def project_1d(self, axis_idx, *axis_ranges, bounds=(None, None)):
        """
        Project multi-dimensional data into one dimention along axis
        with index 'axis_idx'.
        The variable 'axis_ranges' parameter limits the range of all
        other axes, with the position of each axis_range corresponding
        to each axis NOT the axis being projected into.

        For example:

            # projects onto x-axis, y is limited between (1.0, 2.0), z (-1.0, 1.0)
            hist.project_1d(0, (1.0, 2.0), (-1.0, 1.0))

            # projects onto y-axis, x is limited between (1.0, 2.0), z (-1.0, 1.0)
            hist.project_1d(1, (1.0, 2.0), (-1.0, 1.0))

        The optional 'bounds' variable is the limit of the projected
        axis; this defaults to no-limit
        """
        assert 0 <= axis_idx < self.data.ndim

        # merge specified boundries with infinite slice(None) generator
        axis_range = itt.chain(axis_ranges, itt.repeat(slice(None)))

        ranges = []
        summed_axes = []

        for i, axis in enumerate(self.axes):
            if i == axis_idx:
                ranges.append(axis.get_slice(bounds))
            else:
                s = axis.get_slice(next(axis_range))
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
        for i, axis in enumerate(self.axes):
            if i == axis_x:
                ranges.append(axis.get_slice(bounds_x))
            elif i == axis_y:
                ranges.append(axis.get_slice(bounds_y))
            else:
                s = axis.get_slice(next(bounds))
                ranges.append(s)
                if isinstance(s, slice):
                    summed_axes.append(i)

        return self.data[ranges].sum(axis=tuple(summed_axes))

    def __str__(self):
        return '<{dim}D Histogram "{name}" ({sizes}) at {id}>'.format(
            name=self.name,
            dim=self.data.ndim,
            sizes="-".join(map(str, self.data.shape)),
            id="0x%x" % id(self),
        )

    #
    # Math Functions
    #
    def __radd__(self, lhs):
        """
        Rightside add. Applies standard (lefthand) addition, as
        addition is a communitive operation on histograms.
        """
        return self.__add__(lhs)

    def __add__(self, rhs):
        """
        Create a new histogram which is the result of the addition of
        the two histograms.
        The histograms must have the same shape and bins.

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
        self._errors = np.sqrt(self.errors ** 2 + rhs.errors ** 2)
        return self

    def add(self, rhs, scale=1.0):
        """
        Explicit add method, able to automatically scale the right
        hand side.
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
        self._errors = np.sqrt(self.errors ** 2 + rhs.errors ** 2)
        return self

    def __mul__(self, rhs):
        """
        Histogram Multiplication
        """
        hist = copy(self)
        hist *= rhs
        return hist

    def __imul__(self, rhs):
        """
        Histogram Multiplication
        """
        if isinstance(rhs, (int, float)):
            self.data *= rhs
            self._errors *= rhs
        elif isinstance(rhs, np.ndarray):
            self.data *= rhs
            self._errors *= rhs
        else:
            return NotImplemented

        return self

    def __truediv__(self, rhs):
        """
        Histogram Division

        If right hand side is a number, this simply scales the data
        and errors in the histogram.

        If right hand side is another histogram, this will do
        bin-by-bin division, calculating errors appropriately.
        """
        if isinstance(rhs, Histogram):
            quotient = copy(self)
            quotient /= rhs
        elif isinstance(rhs, float):
            quotient = copy(self)
            # quotient.Scale(1.0 / rhs)
            quotient.data /= rhs
        elif isinstance(rhs, np.ndarray):
            quotient = copy(self)
            quotient /= rhs
        else:
            return NotImplemented
        return quotient

    def __itruediv__(self, rhs):
        """
        Inplace histogram division.
        """
        if isinstance(rhs, Histogram):
            num_sq, den_sq = self.data ** 2, rhs.data ** 2
            num_e_sq, den_e_sq = self.errors ** 2, rhs.errors ** 2
            self._errors = np.sqrt(num_e_sq * den_sq + den_e_sq * num_sq)
            with np.errstate(divide='ignore', invalid='ignore'):
                np.divide(self._errors, den_sq, out=self._errors)
                self._errors[np.isfinite(self._errors) != True] = 0
                self.data /= rhs.data
                self.data[np.isfinite(self.data) != True] = 0
        elif isinstance(rhs, float):
            self.data /= rhs
            self._errors /= rhs
        elif isinstance(rhs, np.ndarray):
            assert np.shape(rhs) == np.shape(self.data), "%s != %s" % (np.shape(rhs), np.shape(self.data))
            self.data /= rhs
            self._errors /= rhs
        else:
            return NotImplemented

        return self

    def __matmul__(self, rhs):
        """
        Matrix multiplication
        """
        res = copy(self)
        # res.data = self.data @ rhs
        np.matmul(self.data, rhs, out=res.data)
        return res

    def __rmatmul__(self, lhs):
        """
        Right hand side matrix multiplication
        """
        res = copy(self)
        # res.data = lhs @ self.data
        np.matmul(lhs, self.data, out=res.data)
        return res

    def triple_at(self, index):
        """
        Helper method for yielding a particular (x, y, e) triple in
        the hist
        """
        return self.axes[0][index], self.data[index], self.errors[index]

    def triples(self):
        """
        Helper method for yielding all (x, y, e) values in histogram.
        """
        for index in range(len(self.data)):
            yield self.axes[0][index], self.data[index], self.errors[index]

    @property
    def x_axis(self):
        """
        Returns the first (zeroth) axis in histogram.
        """
        return self.axes[0]

    @property
    def y_axis(self):
        """
        Returns the second axis in histogram.
        """
        return self.axes[1]

    @property
    def z_axis(self):
        """
        Returns the third axis in histogram.
        """
        return self.axes[2]

    def nth_axis(self, idx):
        """
        Returns the nth axis in histogram - zero based.
        Simply access this from the axis tuple.
        """
        return self.axes[idx]


class HistogramRatioPair:
    """
    A numerator-denominator pair of histograms.
    """

    def __init__(self, num, den):
        assert num.axes == den.axes
        self.numerator = num
        self.denominator = den
        self.axes = num.axes
        self._ratio = None

    @classmethod
    def WithKeysInRootObject(cls, container, num_key, den_key):
        keys = (num_key, den_key)
        objs = tuple(get_root_object(container, key) for key in keys)
        for key, obj in zip(keys, objs):
            if obj == None:
                raise ValueError("Histogram identified by %s not found" % key)
        n, d = map(Histogram.BuildFromRootHist, objs)
        return cls(n, d)

    def with_sliced_domain(self, *domain):
        """
        Return a HistogramRatioPair with the domain.
        """
        num, den = self.pair
        domain_slice = self.axes.get_slice(*domain)

        result = self.__new__(self.__class__)
        result.numerator = num.apply_slice(*domain)
        assert result.numerator.axes.shape == result.numerator.data.shape,\
            "%s ≠ %s" % (result.numerator.axes.shape, result.numerator.data.shape)
        result.denominator = den.apply_slice(*domain)
        result.axes = self.axes.masked_by(*domain_slice)
        assert result.axes.shape == result.numerator.data.shape,\
            "%s ≠ %s" % (result.axes.shape, result.numerator.data.shape)
        result._ratio = None
        return result

    def with_masked_zeros(self):
        """
        Return a HistogramRatioPair with the domain.
        """
        num, den = self.pair
        mask_zeros = (num.data != 0.0) & (den.data != 0.0)
        if np.all(mask_zeros):
            return self
        result = self.__new__(self.__class__)
        result.numerator = num.apply_mask(mask_zeros)
        result.denominator = den.apply_mask(mask_zeros)
        result.axes = self.axes.masked_by(mask_zeros)
        assert result.axes.shape == result.numerator.shape,\
            "%s ≠ %s" % (result.axes.shape, result.numerator.shape)
        result._ratio = None
        return result

    @property
    def pair(self):
        """
        Return (numerator, denominator) pair
        """
        return self.numerator, self.denominator

    @property
    def data(self):
        """
        Return numerator & denominator data arrays
        """
        return self.numerator.data, self.denominator.data

    @property
    def ratio(self):
        """
        The ratio of the two histograms
        """
        if self._ratio is None:
            self._ratio = self.numerator / self.denominator
        return self._ratio

    @property
    def shape(self):
        return self.axes.shape

    @property
    def x_axis(self):
        return self.axes[0]

    @property
    def y_axis(self):
        return self.axes[1]

    @property
    def z_axis(self):
        return self.axes[2]

    @property
    def meshgrid(self):
        return self.axes.meshgrid()

    def __iter__(self):
        """
        Yields the numerator followed by denominator.

        This function allows the use of ``num, den = pair`` notation
        """
        yield self.numerator
        yield self.denominator
