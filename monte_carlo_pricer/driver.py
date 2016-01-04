from __future__ import print_function
import sys
import numpy as np
from math import sqrt, exp
from timeit import default_timer as timer
from config import *

def driver(pricer, pinned=False):
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    if pinned:
        from numba import cuda
        with cuda.pinned(paths):
            ts = timer()
            pricer(paths, DT, InterestRate, Volatility)
            te = timer()
    else:
        ts = timer()
        pricer(paths, DT, InterestRate, Volatility)
        te = timer()

    ST = paths[:, -1]
    PaidOff = np.maximum(paths[:, -1] - StrikePrice, 0)
    print('Result')
    fmt = '%20s: %s'
    print(fmt % ('stock price', np.mean(ST)))
    print(fmt % ('standard error', np.std(ST) / sqrt(NumPath)))
    print(fmt % ('paid off', np.mean(PaidOff)))
    optionprice = np.mean(PaidOff) * exp(-InterestRate * Maturity)
    print(fmt % ('option price', optionprice))

    print('Performance')
    NumCompute = NumPath * NumStep
    print(fmt % ('Mstep/second', '%.2f' % (NumCompute / (te - ts) / 1e6)))
    print(fmt % ('time elapsed', '%.3fs' % (te - ts)))

    if '--plot' in sys.argv:
        from matplotlib import pyplot
        pathct = min(NumPath, 100)
        for i in xrange(pathct):
            pyplot.plot(paths[i])
        print('Plotting %d/%d paths' % (pathct, NumPath))
        pyplot.show()

