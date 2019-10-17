# distributions.py
# Matthew Prelee

__all__ = ['rcumsum', 'pmf', 'cmf', 'rcmf', 'median']

import numpy as np
import pandas as pd

def rcumsum(x: pd.Series) -> pd.Series:
    """Reverse cumulative sum of a data Series..

    :param x: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- reverse cumulative distribution

    >>> x = pd.Series(range(5))
    >>> rcumsum(x)
    0    10
    1    10
    2     9
    3     7
    4     4
    dtype: int64
    """

    return np.cumsum(x[::-1])[::-1]


def pmf(x: pd.Series) -> pd.Series:
    """Probability mass function.

    :param x: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- x normalized by sum(x)

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> pmf(x)
    0    0.2
    1    0.2
    2    0.2
    3    0.2
    4    0.2
    dtype: float64
    """

    return x/np.sum(x)


def cmf(x: pd.Series) -> pd.Series:
    """Cumulative mass function.

    :param x: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- x normalized by sum(x) and summed

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> cmf(x)
    0    0.2
    1    0.4
    2    0.6
    3    0.8
    4    1.0
    dtype: float64

    >>> x = pd.Series(data=[5,6,7,8], index=range(4))
    >>> cmf(x)
    0    0.192308
    1    0.423077
    2    0.692308
    3    1.000000
    dtype: float64
    """

    return np.cumsum(pmf(x))


def rcmf(x: pd.Series) -> pd.Series:
    """Reverse cumulative mass function.

    :param x: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- x normalized and then reverse cumulative sum.

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> rcmf(x)
    0    1.0
    1    0.8
    2    0.6
    3    0.4
    4    0.2
    dtype: float64

    >>> x = pd.Series(data=[5,6,7,8], index=range(4))
    >>> rcmf(x)
    0    1.000000
    1    0.807692
    2    0.576923
    3    0.307692
    dtype: float64

    """

    return rcumsum(pmf(x))


def median(x: pd.Series) -> float:
    """Median of weighted value counts.

    :param x: weighted value counts.
    :type : pandas.Series
    :returns:  float -- median of x.index weighted by x.values

    >>> x = pd.Series([1,1,1], index=range(3))
    >>> median(x)
    1.0

    >>> x = pd.Series([9,1], index=range(2))
    >>> median(x)
    0.0

    >>> x = pd.Series([1,1,1,1,1,1], index=range(6))
    >>> median(x)
    2.5

    >>> x = pd.Series([3,4,3,4], index=range(4))
    >>> median(x)
    1.5

    """
    tmp = 2*np.cumsum(x) - sum(x)
    idx_left  = np.searchsorted(tmp, 0, side='left')[0]
    idx_right = np.searchsorted(tmp, 0, side='right')[0]
    if idx_left == idx_right:
        return float(idx_left)
    else:
        return (idx_left + idx_right)/2


if __name__ == '__main__':
    import doctest
    doctest.testmod()

