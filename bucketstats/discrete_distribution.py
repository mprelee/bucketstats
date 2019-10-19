# discrete_distribution.py

import pandas as pd

def _assert_valid_hist(hist: pd.Series):
    """Ensure histogram is numeric and sorted in increasing order."""
    assert isinstance(hist, pd.Series), "Expected pd.Series; got {}".format(type(hist))
    assert hist.index.is_numeric(), "Expected numeric Index; got {}".format(hist.index)
    assert hist.index.is_monotonic_increasing, "Expected monotonic increasing Index; got {}".format(str(hist.index))

def _safe_divide(numer: pd.Series, denom: pd.Series, fill) -> pd.Series:
    """
    Safely divide two series by replacing zero denom with specified fill value

    >>> numer = pd.Series([1,1,1])
    >>> denom = pd.Series([0,1,2])
    >>> fill = 0
    >>> safe_divide(numer, denom, fill)
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
        _assert_valid_hist(hist)
        self._hist = hist

    @property
    def index(self): return self._hist.index

    @property
    def values(self): return self._hist.values

    @property
    def cmf(self): 
        pass

    @property
    def median(self): 
        pass

    @property
    def mean(self): 
        pass

    @property
    def pmf(self): 
        pass

    @property
    def rcmf(self): 
        pass


if __name__ == '__main__':
    x = pd.Series([1,2],[3,4])
    y = DiscreteDistribution(x)
    print(y)
    print(y.cmf)
    print(y.median)

