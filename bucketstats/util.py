import numpy as np
import pandas as pd

__all__ = ['safe_divide']

def safe_divide(numer: pd.Series, denom: pd.Series, fill) -> pd.Series:
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

    result = numer/denom
    result[denom==0] = fill
    return result

def _safe_get_item(x,idx=0):
    """
    >>> _safe_get_item(1)
    1
    >>> _safe_get_item((1,2))
    1
    """
    if '__getitem__' in dir(x):
        return x[idx]
    else:
        return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()

