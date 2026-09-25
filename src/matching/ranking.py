"""Vectorised group statistics shared by the feature and model stages."""
import numpy as np
import pandas as pd


def second_largest(values, groups):
    """Per-row second-largest value of its group (0 for groups of one), without Python loops."""
    v = np.asarray(values, dtype=np.float64)
    codes = pd.factorize(np.asarray(groups))[0]
    order = np.lexsort((-v, codes))
    sc, sv = codes[order], v[order]
    start = np.r_[True, sc[1:] != sc[:-1]]
    first_pos = np.flatnonzero(start)
    size = np.diff(np.r_[first_pos, len(sv)])
    second = np.where(size > 1, sv[np.minimum(first_pos + 1, len(sv) - 1)], 0.0)
    return second[codes]
