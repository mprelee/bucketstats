# distributions.py
# Matthew Prelee

import numpy as np
import pandas as pd

from bucketstats.util import assert_valid_hist

def rcumsum(hist: pd.Series) -> pd.Series:
    """Reverse cumulative sum of a data Series..

    :param hist: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- reverse cumulative distribution

    >>> hist = pd.Series(range(5))
    >>> rcumsum(hist)
    0    10
    1    10
    2     9
    3     7
    4     4
    dtype: int64
    """

    assert_valid_hist(hist)
    return np.cumsum(hist[::-1])[::-1]


def pmf(hist: pd.Series) -> pd.Series:
    """Probability mass function.

    :param hist: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- hist normalized by sum(hist)

    >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> pmf(hist)
    0    0.2
    1    0.2
    2    0.2
    3    0.2
    4    0.2
    dtype: float64
    """

    assert_valid_hist(hist)
    return hist / np.sum(hist)


def cmf(hist: pd.Series) -> pd.Series:
    """Cumulative mass function.

    :param hist: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- hist normalized by sum(hist) and summed

    >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> cmf(hist)
    0    0.2
    1    0.4
    2    0.6
    3    0.8
    4    1.0
    dtype: float64

    >>> hist = pd.Series(data=[5,6,7,8], index=range(4))
    >>> cmf(hist)
    0    0.192308
    1    0.423077
    2    0.692308
    3    1.000000
    dtype: float64
    """

    assert_valid_hist(hist)
    return np.cumsum(pmf(hist))


def rcmf(hist: pd.Series) -> pd.Series:
    """Reverse cumulative mass function.

    :param hist: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- hist normalized and then reverse cumulative sum.

    >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> rcmf(hist)
    0    1.0
    1    0.8
    2    0.6
    3    0.4
    4    0.2
    dtype: float64

    >>> hist = pd.Series(data=[5,6,7,8], index=range(4))
    >>> rcmf(hist)
    0    1.000000
    1    0.807692
    2    0.576923
    3    0.307692
    dtype: float64

    """

    assert_valid_hist(hist)
    return rcumsum(pmf(hist))


def median(hist: pd.Series) -> float:
    """Median of weighted value counts.

    :param hist: weighted value counts.
    :type : pandas.Series
    :returns:  float -- median of hist.index weighted by hist.values

    >>> hist = pd.Series([1,1,1], index=range(3))
    >>> median(hist)
    1.0

    >>> hist = pd.Series([9,1], index=range(2))
    >>> median(hist)
    0.0

    >>> hist = pd.Series([1,1,1,1,1,1], index=range(6))
    >>> median(hist)
    2.5

    >>> hist = pd.Series([3,4,3,4], index=range(4))
    >>> median(hist)
    1.5

    """

    assert_valid_hist(hist)
    tmp = 2 * np.cumsum(hist) - sum(hist)
    _idx_left = np.searchsorted(tmp, 0, side='left').tolist()
    _idx_right = np.searchsorted(tmp, 0, side='right').tolist()
    idx_left = _idx_left if not isinstance(_idx_left, list) else _idx_left[0]
    idx_right = _idx_right if not isinstance(_idx_right, list) else _idx_right[0]
    if idx_left == idx_right:
        return float(idx_left)
    else:
        return (idx_left + idx_right) / 2
