# survival_analysis.py
# Matthew Prelee

import numpy as np
import pandas as pd
from probability import *
from util import safe_divide


def survival(x: pd.Series) -> pd.Series:
    """Survival function.  Alias for reverse cumulative mass function.

    :param x: binned histogram.
    :type : pandas.Series
    :returns:  pandas.Series -- x normalized and then reverse cumulative sum.

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> survival(x)
    0    1.0
    1    0.8
    2    0.6
    3    0.4
    4    0.2
    dtype: float64

    >>> x = pd.Series(data=[5,6,7,8], index=range(4))
    >>> survival(x)
    0    1.000000
    1    0.807692
    2    0.576923
    3    0.307692
    dtype: float64
    """

    return rcmf(x)


def _ni(x: pd.Series) -> pd.Series:
    return rcumsum(x)


def _di(x: pd.Series, x_obs: pd.Series) -> pd.Series:
    return x_obs if x_obs is not None else x


def kaplan_meier(x: pd.Series, x_obs: pd.Series = None) -> pd.Series:
    """
    Kaplan-Meier estimate of survival function.
    https://en.wikipedia.org/wiki/Kaplan–Meier_estimator

    :param x: binned histogram.
    :type : pandas.Series
    :param x_obs: observed values.  If None, assumes x=x_obs
    :type : pandas.Series
    :returns:  pandas.Series -- Kaplan-Meier estimate of survival
    function of x

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> kaplan_meier(x)
    0    0.8
    1    0.6
    2    0.4
    3    0.2
    4    0.0
    dtype: float64
    """

    ni = _ni(x)
    di = _di(x, x_obs)
    return np.cumprod(1 - safe_divide(di, ni, 0))


def var_kaplan_meier(x: pd.Series, x_obs: pd.Series = None) -> pd.Series:
    """
    Kaplan-Meier estimate of survival function.
    https://en.wikipedia.org/wiki/Kaplan–Meier_estimator

    :param x: binned histogram.
    :type : pandas.Series
    :param x_obs: observed values.  If None, assumes x=x_obs
    :type : pandas.Series
    :returns:  pandas.Series -- Kaplan-Meier estimate of survival
    function of x

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> var_kaplan_meier(x)
    0    0.512000
    1    0.270000
    2    0.106667
    3    0.020000
    4    0.000000
    dtype: float64
    """

    ni, di = _ni(x), _di(x, x_obs)
    km_est = kaplan_meier(x, x_obs)
    inner = safe_divide(di, ni * (ni - di), 0)
    return km_est**2 * rcumsum(inner)


def nelson_aalen(x: pd.Series, x_obs: pd.Series = None) -> pd.Series:
    """
    Nelson-Aalen estimate of cumulative hazard rate.
    https://en.wikipedia.org/wiki/Nelson–Aalen_estimator

    :param x: binned histogram.
    :type : pandas.Series
    :param x_obs: observed values.  If None, assumes x=x_obs
    :type : pandas.Series
    :returns:  pandas.Series -- Kaplan-Meier estimate of survival
    function of x

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> nelson_aalen(x)
    0    2.283333
    1    2.083333
    2    1.833333
    3    1.500000
    4    1.000000
    dtype: float64

    """

    ni = _ni(x)
    di = _di(x, x_obs)
    return rcumsum(di / ni)


def var_nelson_aalen(x: pd.Series, x_obs: pd.Series = None) -> pd.Series:
    """
    Variance of Nelson-Aalen estimate of cumulative hazard rate.
    https://www.statsdirect.com/help/survival_analysis/kaplan.htm

    :param x: binned histogram.
    :type : pandas.Series
    :param x_obs: observed values.  If None, assumes x=x_obs
    :type : pandas.Series
    :returns:  pandas.Series -- Kaplan-Meier estimate of survival
    function of x

    >>> x = pd.Series(data=[1,1,1,1,1], index=range(5))
    >>> var_nelson_aalen(x)
    0    0.800000
    1    0.750000
    2    0.666667
    3    0.500000
    4    0.000000
    dtype: float64
    """

    numer = var_kaplan_meier(x, x_obs)
    denom = kaplan_meier(x, x_obs)**2
    return safe_divide(numer, denom, 0)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
