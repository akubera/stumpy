#
# test/histogram.py
#

import numpy as np
from stumpy.histogram import Histogram

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
