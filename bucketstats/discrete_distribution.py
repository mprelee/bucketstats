# discrete_distribution.py

import pandas as pd
import numpy as np


def _assert_valid_hist(hist: pd.Series):
    """Ensure histogram is numeric and sorted in increasing order.

    >>> a = "not_a_histogram"
    >>> DiscreteDistribution(a)
    Traceback (most recent call last):
    ...
    AssertionError: Expected pd.Series; got <class 'str'>

    >>> b = pd.Series(data=[1,2,3],index=['a','b','c'])
    >>> DiscreteDistribution(b)
    Traceback (most recent call last):
    ...
    AssertionError: Expected numeric Index; got Index(['a', 'b', 'c'], dtype='object')

     >>> c = pd.Series(data=[1,2,3],index=[4,6,5])
    >>> DiscreteDistribution(c)
    Traceback (most recent call last):
    ...
    AssertionError: Expected monotonic increasing Index; got Int64Index([4, 6, 5], dtype='int64')
    
    """
    assert isinstance(hist, pd.Series), "Expected pd.Series; got {}".format(type(hist))
    assert hist.index.is_numeric(), "Expected numeric Index; got {}".format(hist.index)
    assert hist.index.is_monotonic_increasing, "Expected monotonic increasing Index; got {}".format(str(hist.index))


def _safe_divide(numer: pd.Series, denom: pd.Series, fill) -> pd.Series:
    """
    Safely divide two series by replacing zero denom with specified fill value

    >>> numer = pd.Series([1,1,1])
    >>> denom = pd.Series([0,1,2])
    >>> fill = 0
    >>> _safe_divide(numer, denom, fill)
    0    0.0
    1    1.0
    2    0.5
    dtype: float64
    """

    result = numer / denom
    result[denom == 0] = fill
    return result


class DiscreteDistribution:

    def __init__(self, hist):
        """Constructor for  DiscreteDistribution object.
        The provided histogram must be a pandas Series
        whose Index must be numeric and monotonically increasing 
        (and therefore, sorted)

        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        """
        _assert_valid_hist(hist)
        self._hist = hist

    def toseries(self) -> pd.Series:
        """ Return distribution as a series.  Returns original histogram

        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.toseries()
        0    0
        1    1
        2    2
        3    3
        4    4
        dtype: int64
        """
        return self._hist

    @property
    def index(self) -> pd.Index: 
        """Support of underlying histogram, as a pandas Index.

        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.index
        RangeIndex(start=0, stop=5, step=1)
        """

        return self._hist.index

    @property
    def values(self): 
        """Values of underlying histogram, which are the counts
        of each histogram "bucket" of the index
        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.values
        array([0, 1, 2, 3, 4])
        """

        return self._hist.values

    @property
    def sum(self):
        """Sum of underlying histogram values

        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.sum
        10
        """

        return self._hist.sum()

    @property
    def cumsum(self) -> pd.Series:
        """Cumulative sum of the underlying histogram.
        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.sum
        10
        """
        return np.cumsum(self._hist)

    @property
    def rcumsum(self) -> pd.Series:
        """Reverse cumulative sum of the underlying histogram.
        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.rcumsum
        0    10
        1    10
        2     9
        3     7
        4     4
        dtype: int64
        """
 
        return np.cumsum(self._hist[::-1])[::-1]

    @property
    def cmf(self) -> pd.Series: 
        """Cumulative mass function.

        >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.cmf
        0    0.2
        1    0.4
        2    0.6
        3    0.8
        4    1.0
        dtype: float64

        >>> hist = pd.Series(data=[5,6,7,8], index=range(4))
        >>> d = DiscreteDistribution(hist)
        >>> d.cmf
        0    0.192308
        1    0.423077
        2    0.692308
        3    1.000000
        dtype: float64
        """
        return self.cumsum / self.sum

    @property
    def rcmf(self) -> pd.Series: 
        """Reverse cumulative mass function.

        >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.rcmf
        0    1.0
        1    0.8
        2    0.6
        3    0.4
        4    0.2
        dtype: float64

        >>> hist = pd.Series(data=[5,6,7,8], index=range(4))
        >>> d = DiscreteDistribution(hist)
        >>> d.rcmf
        0    1.000000
        1    0.807692
        2    0.576923
        3    0.307692
        dtype: float64
        """

        return self.rcumsum / self.sum

    @property
    def pmf(self) -> pd.Series: 
        """Probability mass function.

        >>> hist = pd.Series(data=[1,1,1,1,1], index=range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.pmf
        0    0.2
        1    0.2
        2    0.2
        3    0.2
        4    0.2
        dtype: float64
        """
        return self._hist / self.sum

    @property
    def mean(self): 
        """Expected value of distribution
        >>> hist = pd.Series(range(5))
        >>> d = DiscreteDistribution(hist)
        >>> d.mean
        3.0
        """
        return np.inner(self.index, self.pmf)

    @property
    def median(self) -> float: 
        """Median of underlying distribution

        >>> hist = pd.Series([1,1,1], index=range(3))
        >>> d = DiscreteDistribution(hist)
        >>> d.median
        1.0

        >>> hist = pd.Series([9,1], index=range(2))
        >>> d = DiscreteDistribution(hist)
        >>> d.median
        0.0

        >>> hist = pd.Series([1,1,1,1,1,1], index=range(6))
        >>> d = DiscreteDistribution(hist)
        >>> d.median
        2.5

        >>> hist = pd.Series([3,4,3,4], index=range(4))
        >>> d = DiscreteDistribution(hist)
        >>> d.median
        1.5
        """

        tmp = 2 * self.cumsum - self.sum
        _idx_left = np.searchsorted(tmp, 0, side='left').tolist()
        _idx_right = np.searchsorted(tmp, 0, side='right').tolist()
        idx_left = _idx_left if not isinstance(_idx_left, list) else _idx_left[0]
        idx_right = _idx_right if not isinstance(_idx_right, list) else _idx_right[0]
        if idx_left == idx_right:
            return float(idx_left)
        else:
            return (idx_left + idx_right) / 2

if __name__ == '__main__':
    import doctest
    doctest.testmod()

