from __future__ import print_function

import math

import numpy as np

from numba import cuda, vectorize
from accelerate.cuda.rand import PRNG
from cuda_helper import MM


@vectorize(['f8(f8, f8, f8, f8, f8)'], target='cuda')
def step(last, dt, c0, c1, noise):
    return last * math.exp(c0 * dt + c1 * noise)

def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]

    mm = MM(shape=n, dtype=np.double, prealloc=5)

    blksz = cuda.get_current_device().MAX_THREADS_PER_BLOCK
    gridsz = int(math.ceil(float(n) / blksz))

    stream = cuda.stream()
    prng = PRNG(PRNG.MRG32K3A, stream=stream)

    # Allocate device side array
    d_normdist = cuda.device_array(n, dtype=np.double, stream=stream)

    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    d_last = cuda.to_device(paths[:, 0], to=mm.get())
    for j in range(1, paths.shape[1]):
        prng.normal(d_normdist, mean=0, sigma=1)
        d_paths = cuda.to_device(paths[:, j], stream=stream, to=mm.get())
        step(d_last, dt, c0, c1, d_normdist, out=d_paths, stream=stream)
        d_paths.copy_to_host(paths[:, j], stream=stream)
        mm.free(d_last)
        d_last = d_paths

    stream.synchronize()

if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer, pinned=True)
