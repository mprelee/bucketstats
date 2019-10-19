# distributions.py
# Matthew Prelee

import numpy as np
import pandas as pd

#def rcumsum(hist: pd.Series) -> pd.Series:
#    """Reverse cumulative sum of a data Series..
#
#    >>> hist = pd.Series(range(5))
#    >>> rcumsum(hist)
#    0    10
#    1    10
#    2     9
#    3     7
#    4     4
#    dtype: int64
#    """
#    pass
#
#
#def pmf(hist: pd.Series) -> pd.Series:
#    pass
#
#def cmf(hist: pd.Series) -> pd.Series:
#    """Cumulative mass function.
#
#    :param hist: binned histogram.
#    :type : pandas.Series
#    :returns:  pandas.Series -- hist normalized by sum(hist) and summed
#
#    >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
#    >>> cmf(hist)
#    0    0.2
#    1    0.4
#    2    0.6
#    3    0.8
#    4    1.0
#    dtype: float64
#
#    >>> hist = pd.Series(data=[5,6,7,8], index=range(4))
#    >>> cmf(hist)
#    0    0.192308
#    1    0.423077
#    2    0.692308
#    3    1.000000
#    dtype: float64
#    """
#    pass
#
#
#def rcmf(hist: pd.Series) -> pd.Series:
#    """Reverse cumulative mass function.
#
#    :param hist: binned histogram.
#    :type : pandas.Series
#    :returns:  pandas.Series -- hist normalized and then reverse cumulative sum.
#
#    >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
#    >>> rcmf(hist)
#    0    1.0
#    1    0.8
#    2    0.6
#    3    0.4
#    4    0.2
#    dtype: float64
#
#    >>> hist = pd.Series(data=[5,6,7,8], index=range(4))
#    >>> rcmf(hist)
#    0    1.000000
#    1    0.807692
#    2    0.576923
#    3    0.307692
#    dtype: float64
#
#    """
#    pass
#
#
#def median(hist: pd.Series) -> float:
#    """Median of weighted value counts.
#
#    :param hist: weighted value counts.
#    :type : pandas.Series
#    :returns:  float -- median of hist.index weighted by hist.values
#
#    >>> hist = pd.Series([1,1,1], index=range(3))
#    >>> median(hist)
#    1.0
#
#    >>> hist = pd.Series([9,1], index=range(2))
#    >>> median(hist)
#    0.0
#
#    >>> hist = pd.Series([1,1,1,1,1,1], index=range(6))
#    >>> median(hist)
#    2.5
#
#    >>> hist = pd.Series([3,4,3,4], index=range(4))
#    >>> median(hist)
#    1.5
#
#    """
#    pass
#
#
#def mean(hist: pd.Series) -> float:
#    pass
