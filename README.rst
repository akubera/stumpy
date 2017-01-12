======
stumpy
======

A python library for bridging ROOT_ data and numpy_ structures.

The primary goal of stumpy is to create a very *pythonic* interface to the very capable
ROOT and numpy/scipy libraries that already exist.


Style Guide
-----------

Most code is written with the pep8 style consideration.

An exception is for class methods, which uses the ``CapWords`` naming convention
to distinguish themselves from instance methods.


Example Usage
-------------

.. code:: python

    >>> from ROOT import TH1F
    >>> from stumpy import Histogram

    # This would presumably be loaded from a file
    >>> hist = TH1F("hist", "RandomHistogram", 144, -1, 1)
    >>> hist.FillRandom("gaus", 10_000)
    >>> h = Histogram.BuildFromRootHist(hist)
    # imitates numpy array
    >>> h.shape
    (144, )
    # use integer to index by bin
    >>> h[8]
    8.0
    # index by bin range
    >>> h[50:60]
    array([ 6., 9., 7., 6., 4., 12., 7., 5., 7, 13.], dtype=float32]])
    # use float to index by coordinate
    >>> h[0.8]
    2.0
    # in what bin number do we find 0.8?
    >>> h.bin_at(0.8)
    129
    # check it
    >>> h[129]
    2.0
    # center-coordinate value of bin 129
    >>> h.x_axis.bin_centers[129]
    0.798611111111111
    # mix and match indexing - go from 65th bin to the bin containing 0
    >>> h[65:0.0]
    array([8., 12., 10., 5., 9., 12., 13.], dtype=float32)
    # NOTE! stumpy bin numbering uses zero-based indexes, not 1 like ROOT
    >>> h[7] == hist.GetBinContent(7)
    False
    >>> all(h[i] == hist.GetBinContent(i + 1) for i in range(144))
    True

    # Histograms have a fill function imitating the ROOT histograms
    >>> h.fill(0.8)
    (129, )
    >>> h[0.8]
    3.0
    # fill_all() is used to apply fill() to items in a collection
    >>> h.fill_all([0.8, 0.9, -.045])

    # don't forget errors
    >>> h.errors[60]
    2.64575131


License
-------

Stumpy source is released under the terms of the `New BSD License`_.



.. _ROOT: https://root.cern.ch/
.. _numpy: http://www.numpy.org/

.. _New BSD License: https://opensource.org/licenses/BSD-3-Clause
