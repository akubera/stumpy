#
# stumpy/axis.py
#
"""
Submodule containing the Axis, BinnedAxis, and AxisCollection classes.
"""

import numpy as np
from itertools import zip_longest as _zip_longest
from enum import Enum

Overflow = type("OVERFLOW", (), {})()
Underflow = type("UNDERFLOW", (), {})()


class ByEdge(Enum):
    High = 0
    Low = 1
    Center = 2

    @classmethod
    def From(cls, name):
        return cls[name[0].upper() + ''.join(map(str.lower, name[1:]))]


class Axis:
    """
    Histogram axis class. Contains all information about binning, including
    labels. Imitates the ROOT class TAxis.

    Unlike the ROOT class, this class provides a method for determining
    whether the associated values of each bin should be interpreted to be
    the low edge, high edge, or center.
    """

    OVERFLOW = Overflow
    UNDERFLOW = Underflow

    ndim = 1

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
        title : str
            Optional title for the axis
        count_min_max : tuple
            Tuple containing the number of bins, the low x value, and the
            high x value.
        """
        self.title = kwargs.pop('title', '')

        if data.__class__.__name__.startswith(('TH', 'TAxis')):
            raise TypeError("Cannot construct Axis from ROOT object. Use "
                            "class method 'BuildFromRootAxis'.")

        if sum(key in ('low_edges', 'centers', 'high_edges') for key in kwargs.keys()) > 1:
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
        elif isinstance(data, Axis):
            self._low_edges = data._low_edges
            self._bin_width = self._low_edges[1] - self._low_edges[0]
            if not self.title:
                self.title = data.title
        else:
            self._low_edges = data
            self._bin_width = self._low_edges[1] - self._low_edges[0]

        self._xmin = self._low_edges[0]
        try:
            self._xmax = self._low_edges[-1] + self._bin_width
        except IndexError:
            print(':-(', self._low_edges)
            raise

    #
    # Alternate Constructors
    #

    @classmethod
    def BuildWithLinearSpacing(cls, nbins, min_x, max_x, **kwargs):
        self = cls(count_min_max=(nbins, min_x, max_x,), **kwargs)
        return self

    @classmethod
    def FromROOTAxis(cls, axis):
        nbins = axis.GetNbins()
        if axis.IsVariableBinSize():
            bin_array = np.frombuffer(axis.GetXbins(), dtype='f8', count=nbins)
            self = cls(bin_array)
        else:
            self = cls.BuildWithLinearSpacing(nbins,
                                              axis.GetXmin(),
                                              axis.GetXmax())
        self.title = axis.GetTitle()
        return self

    def sliced_by(self, slice_):
        """
        Create axis with a slice applied to the data (_low_edges)
        """
        slice_ = self.get_slice(slice_)
        axis = self.masked_by(slice_)
        return axis

    def masked_by(self, mask):
        """
        Create axis with a masked applied to the data (_low_edges)
        """
        axis = self.__new__(self.__class__)
        axis._low_edges = self._low_edges[mask]
        axis._bin_width = axis._low_edges[1] - axis._low_edges[0]
        axis.title = self.title
        return axis

    #
    # Properties
    #

    @property
    def min(self):
        return self._low_edges[0]

    @property
    def max(self):
        return self._low_edges[-1] + self._bin_width

    @property
    def bounds(self):
        """
        Return (min, max) pair
        """
        return self.min, self.max

    @property
    def nbins(self):
        return len(self._low_edges)

    @property
    def bin_centers(self):
        try:
            return self._centers
        except AttributeError:
            self._centers = self._low_edges + self._bin_width / 2.0
            return self._centers

    @property
    def shape(self):
        return np.shape(self._low_edges)

    #
    # Data access
    #

    def __getitem__(self, index):
        """
        Returns the value of bin specified by index
        """
        return self.bin_centers[index]

    def bin_at(self, value):
        return self.getbin(value)

    def getbin(self, value):
        """
        Return the bin relating to value. Bin counting starts at 0.
        """
        if isinstance(value, float):
            if value < self._xmin:
                return Underflow
            elif value >= self._xmax:
                return Overflow
            else:
                idx = np.searchsorted(self._low_edges, value, side='right') - 1
                return idx
        if isinstance(value, slice):
            start = self.getbin(value.start)
            stop = self.getbin(value.stop)
            return slice(start, stop)
        if isinstance(value, (tuple, list)):
            start, stop = map(self.getbin, value)
            if isinstance(stop, int):
                stop += 1
            return slice(start, stop)

        return value

    def get_slice(self, value):
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
            return np.copy(self._low_edges)
        else:
            return np.linspace(n, self._low_edges[0], self._low_edges[-1])

    def bounded_domain(self, value):
        """
        Return a numpy array containing the bin centers of the range that
        this axis' domain.
        """
        s = self.get_slice(value)
        return self.bin_centers[s]

    def __eq__(self, rhs):
        if isinstance(rhs, Axis):
            return np.all(self._low_edges == rhs._low_edges)
        return NotImplemented

    def __len__(self):
        return len(self._low_edges)


class MultiAxis(tuple, Axis):
    """
    Pseudo-axis used for multiple dimension data.
    """

    @classmethod
    def FromData(cls, data):
        """
        Create a MultiAxis object from aritrary objects
        """
        def matches_bininfo_pattern(obj):
            numtype = (float, int)
            if isinstance(obj, tuple) and len(obj) == 3:
                return (type(obj[0]) == int and
                        type(obj[1]) in numtype and
                        type(obj[2]) in numtype)
            return False

        def data_to_axes(obj):
            if isinstance(obj, Axis):
                yield obj
            elif isinstance(obj, np.ndarray):
                yield Axis(obj)
            elif isinstance(obj, tuple) and len(obj) == 1:
                yield Axis(obj[0])
            elif matches_bininfo_pattern(obj):
                yield Axis.BuildWithLinearSpacing(*obj)
            else:
                try:
                    it = iter(obj)
                except TypeError:
                    yield Axis(obj)
                else:
                    for n in it:
                        yield from data_to_axes(n)
                    # elif isinstance(axis_parameters[0], int):
                    #     newaxis =
                    #     axis_kwargs = kwargs.get('x_params', {})
                    #     axis_kwargs.update(kwargs.get('axis_params', ({}, ))[0])
                    #     yield from (Axis.BuildWithLinearSpacing(*axis_parameters, **axis_kwargs), ))
                    # else:
                    #     self.axes = MultiAxis(*map(Axis, axis_parameters))
        self = cls.__new__(cls, tuple(data_to_axes(data)))
        self._axes = self
        return self

    @classmethod
    def FromRootHistogram(cls, hist):
        """
        Create a MultiAxis object from axes found in a ROOT histogram
        """
        from ROOT import TH1
        if not isinstance(hist, TH1):
            raise TypeError("Expected ROOT histogram, found %r " % hist)

        hist_axes = (hist.GetXaxis(), hist.GetYaxis(), hist.GetZaxis())
        axes = tuple(map(Axis.FromROOTAxis, hist_axes[:hist.GetDimension()]))

        self = cls.__new__(cls, axes)
        self._axes = self
        return self

    def get_bin(self, value):
        if len(value) != len(self._axes):
            raise ValueError("Dimension of value does not equal dimension of axes! "
                             "(%d > %d)" % (len(value), len(self._axes)))
        return tuple(a.get_slice(s) for a, s in _zip_longest(self._axes, value))

    def get_slice(self, *slices):
        if len(slices) > len(self._axes):
            raise RuntimeError("More slices than axes! "
                               "(%d > %d)" % (len(slices), len(self._axes)))
        return tuple(a.get_slice(s) for a, s in _zip_longest(self._axes, slices))

    def sliced_by(self, *slices):
        """
        Construct new MultiAxis using sliced versions of associated axes.
        """
        cls = self.__class__
        if len(slices) > len(self._axes):
            raise RuntimeError("More slices than axes! "
                               "(%d > %d)" % (len(slices), len(self._axes)))

        axes = tuple(a.sliced_by(s) for a, s in _zip_longest(self._axes, slices))

        multiaxis = cls.__new__(cls, axes)
        multiaxis._axes = multiaxis
        return multiaxis

    def masked_by(self, *masks):
        """
        Construct new MultiAxis using masked versions of associated axes.
        """
        cls = self.__class__
        axes = tuple(a.masked_by(m) for a, m in _zip_longest(self._axes, masks))

        multiaxis = cls.__new__(cls, axes)
        multiaxis._axes = multiaxis
        return multiaxis

    @property
    def shape(self):
        return tuple(len(a) for a in self)

    @property
    def ndim(self):
        return len(self._axes)

    @property
    def bin_info(self):
        """
        Return flattened list of (bin count, min, max) triples
        for use in constructing ROOT histograms
        """
        result = []
        for a in self._axes:
            result += [int(a.nbins), float(a.min), float(a.max)]
        return result

    def meshgrid(self, point_at='center'):
        """
        Return numpy 'meshgrid' parameters created from all axes.

        Parameters
        ----------
            point_at : string
                identifier of
        """
        from operator import attrgetter
        bins = {
            'low': attrgetter('_low_edges'),
            'center': attrgetter('bin_centers'),
            'high': attrgetter('_high_edges'),
        }[point_at]
        return np.meshgrid(*map(bins, self._axes), indexing='ij')

    # def __iter__(self):
    #     return super().__iter__()

    def __getitem__(self, key):
        return self.axis(key)

    def axis(self, name):
        if not isinstance(name, int):
            name = self.axis_choice(name)
        return super().__getitem__(name)

    @staticmethod
    def axis_choice(name):
        return {
            'X': 1,
            'Y': 2,
            'Z': 3,
        }.get(name.upper(), 0)
