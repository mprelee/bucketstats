# discrete_distribution.py

import pandas as pd
from util import assert_valid_hist
import probability as p


class DiscreteDistribution(pd.Series):

    def __init__(self, *args, **kwargs):
        super(DiscreteDistribution, self).__init__(*args, **kwargs)
        assert_valid_hist(self)

    @property
    def cmf(self): return p.cmf(self)

    @property
    def median(self): return p.median(self)

    @property
    def pmf(self): return p.pmf(self)

    @property
    def rcmf(self): return p.rcmf(self)



if __name__ == '__main__':
    x = pd.Series([1,2],[3,4])
    y = DiscreteDistribution(x)
    print(y)
    print(y.cmf)
    print(y.median)

