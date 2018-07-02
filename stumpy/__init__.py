#
# stumpy/__init__.py
#
"""
"""

from .histogram import Histogram, HistogramRatioPair
from .utils import get_root_object
import numpy as np


def chi2(data, model):
    """
    Returns
    """
    return np.sum((data - model) ** 2 / model ** 2)
