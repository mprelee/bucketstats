# discrete_distribution.py

import pandas as pd
from bucketstats.util import assert_valid_hist


class DiscreteDistribution(pd.Series):

    def __init__(self, *args, **kwargs):
        super(DiscreteDistribution, self).__init__(*args, **kwargs)
        assert_valid_hist(self)


if __name__ == '__main__':
    x = pd.Series([1,2],[3,4])
    y = DiscreteDistribution(x)
    print(y)
    a = pd.Series([1,2],['a','b'])
    b = DiscreteDistribution(a)

