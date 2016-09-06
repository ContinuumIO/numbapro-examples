from __future__ import print_function

import math

import numpy as np

from numba import cuda, vectorize
from accelerate.cuda.rand import PRNG


@vectorize(['f8(f8, f8, f8, f8, f8)'], target='cuda')
def step(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]

    blksz = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    gridsz = int(math.ceil(float(n) / blksz))

    # Instantiate cuRAND PRNG
    prng = PRNG(PRNG.MRG32K3A)

    # Allocate device side array
    d_normdist = cuda.device_array(n, dtype=np.double)

    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    # Simulation loop
    d_last = cuda.to_device(paths[:, 0])
    for j in range(1, paths.shape[1]):
        prng.normal(d_normdist, mean=0, sigma=1)
        d_paths = cuda.to_device(paths[:, j])
        step(d_last, dt, c0, c1, d_normdist, out=d_paths)
        d_paths.copy_to_host(paths[:, j])
        d_last = d_paths

if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer)
