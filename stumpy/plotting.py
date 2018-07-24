#
# stumpy/plotting.py
#


def get_projections(hist, p=None):
    """
    Symmetric projections of multi-dim histogram

    Parameters
    ----------
    hist : Histogram
        histograms

    p : projection slice
        If none, takes either middle-most bin, or full


    Yeilds
    ------
    1D histograms

    """

    if p is None:
        p = ()

    if len(p) == 2:
        p *= 2


    from ROOT import TH3
    p_funcs = (TH3.ProjectionX, TH3.ProjectionY, TH3.PRojectionZ)

    for pfunc in p_funcs:
        h = pfunc(hist, 'name', *p)
        h.SetStats(False)
        yield h



def hist_get_projection(num, den):
    ns = get_projections(num)
    ds = get_projections(den)

    for n, d in zip(ns, ds):
        n.Divide(d)
        yield n

