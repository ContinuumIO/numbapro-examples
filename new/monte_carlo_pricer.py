"""
A Monte Carlo Pricer

Demonstrated features:
* Numba Vectorize
* NumbaPro CUDA Vectorize
* NumbaPro cuRAND

"""

from __future__ import print_function, division

import sys
import datetime
import math
from math import sqrt, exp
from timeit import default_timer as timer

import numpy as np

import numba
from numba import cuda, vectorize
from numbapro.cudalib import curand

from bokeh.plotting import figure, show, output_notebook

output_notebook()

"""
**Version information:**
"""

print("This file is generated on:", datetime.datetime.now())
print("python: {0}.{1}".format(*sys.version_info[:2]))
print("numpy:", np.__version__)
print("numba:", numba.__version__)
print("CUDA GPU:", cuda.gpus[0].name)

"""
## Setup
"""

"""
Setup constants for the simulator
"""

StockPrice = 20.83
StrikePrice = 21.50
Volatility = 0.021  # per year
InterestRate = 0.20
Maturity = 5. / 12.
NumPath = 500000
NumStep = 200

"""
A simulation driver that uses different pricer implementation
"""


def driver(pricer, pinned=False):
    paths = np.zeros((NumPath, NumStep + 1), order='F')
    paths[:, 0] = StockPrice
    DT = Maturity / NumStep

    if pinned:
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

    # Plot
    pathct = min(NumPath, 100)
    fig = figure()

    for i in range(pathct):
        fig.line(np.arange(paths[i].size), paths[i])
    print('Plotting %d/%d paths' % (pathct, NumPath))
    show(fig)
    return (te - ts)


"""
## A NumPy Version
"""


def numpy_step(dt, prices, c0, c1, noises):
    return prices * np.exp(c0 * dt + c1 * noises)


def numpy_mcp(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in range(1, paths.shape[1]):
        prices = paths[:, j - 1]
        noises = np.random.normal(0., 1., prices.size)
        paths[:, j] = numpy_step(dt, prices, c0, c1, noises)


"""
Testing
"""

numpy_runtime = driver(numpy_mcp)

"""
## A Numba Vectorize Version
"""

function_signature = ['float64(float64, float64, float64, float64, float64)']


@vectorize(function_signature, target='cpu')
def numba_step(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)


def numba_mcp(paths, dt, interest, volatility):
    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * np.sqrt(dt)

    for j in range(1, paths.shape[1]):
        prices = paths[:, j - 1]
        noises = np.random.normal(0., 1., prices.size)
        numba_step(prices, dt, c0, c1, noises, out=paths[:, j])


"""
Testing
"""

numba_runtime = driver(numba_mcp)

"""
## A NumbaPro CUDA Version

Uses cuRAND for on GPU random number generation
"""


@vectorize(function_signature, target='cuda')
def numbapro_cuda_step(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)


def numbapro_cuda_mcp(paths, dt, interest, volatility):
    n = paths.shape[0]

    # Instantiate cuRAND PRNG
    prng = curand.PRNG(curand.PRNG.MRG32K3A)

    # Allocate device side array
    d_normdist = cuda.device_array(n, dtype=np.double)

    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    # Simulation loop
    d_last = cuda.to_device(paths[:, 0])
    for j in range(1, paths.shape[1]):
        prng.normal(d_normdist, mean=0, sigma=1)
        d_paths = cuda.to_device(paths[:, j])
        numbapro_cuda_step(d_last, dt, c0, c1, d_normdist, out=d_paths)
        d_paths.copy_to_host(paths[:, j])
        d_last = d_paths


"""
Testing
"""

cuda_runtime = driver(numbapro_cuda_mcp)

"""
Speedups
"""

print("Numba CPU speedup over NumPy:", numpy_runtime / numba_runtime)
print("NumbaPro CUDA speedup over Numba CPU:", numba_runtime / cuda_runtime)
