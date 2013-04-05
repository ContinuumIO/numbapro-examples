import numpy as np
from math import sqrt, exp
from timeit import default_timer as timer
#from matplotlib import pyplot

from config import *
def driver(pricer):
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    ts = timer()
    pricer(paths, DT, InterestRate, Volatility)
    te = timer()

    ST = paths[:, -1]
    PaidOff = np.maximum(paths[:, -1] - StrikePrice, 0)
    print 'Result'
    fmt = '%20s: %s'
    print fmt % ('stock price', np.mean(ST))
    print fmt % ('standard error', np.std(ST) / sqrt(NumPath))
    print fmt % ('paid off', np.mean(PaidOff))
    optionprice = np.mean(PaidOff) * exp(-InterestRate * Maturity)
    print fmt % ('option price', optionprice)

    print 'Performance'
    NumCompute = NumPath * NumStep
    print fmt % ('Mstep/second', '%.2f' % (NumCompute / (te - ts) / 1e6))
    print fmt % ('time elapsed', '%.3fs' % (te - ts))


#    for i in xrange(NumPath):
#        pyplot.plot(paths[i])
#    pyplot.show()

