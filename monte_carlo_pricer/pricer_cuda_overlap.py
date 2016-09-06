#! /usr/bin/env python
"""
This version demonstrates copy-compute overlapping through multiple streams.
"""
from __future__ import print_function

import math
import sys

import numpy as np

from numba import cuda, jit
from accelerate.cuda.rand import PRNG

from cuda_helper import MM

if sys.version_info[0] == 2:
    range = xrange

@jit('void(double[:], double[:], double, double, double, double[:])',
     target='cuda')
def cu_step(last, paths, dt, c0, c1, normdist):
    i = cuda.grid(1)
    if i >= paths.shape[0]:
        return
    noise = normdist[i]
    paths[i] = last[i] * math.exp(c0 * dt + c1 * noise)

def monte_carlo_pricer(paths, dt, interest, volatility):
    n = paths.shape[0]
    num_streams = 2

    part_width = int(math.ceil(float(n) / num_streams))
    partitions = [(0, part_width)]
    for i in range(1, num_streams):
        begin, end = partitions[i - 1]
        begin, end = end, min(end + (end - begin), n)
        partitions.append((begin, end))
    partlens = [end - begin for begin, end in partitions]

    mm = MM(shape=part_width, dtype=np.double, prealloc=10 * num_streams)

    device = cuda.get_current_device()
    blksz = device.MAX_THREADS_PER_BLOCK
    gridszlist = [int(math.ceil(float(partlen) / blksz))
                  for partlen in partlens]

    strmlist = [cuda.stream() for _ in range(num_streams)]

    prnglist = [PRNG(PRNG.MRG32K3A, stream=strm)
                for strm in strmlist]

    # Allocate device side array
    d_normlist = [cuda.device_array(partlen, dtype=np.double, stream=strm)
                  for partlen, strm in zip(partlens, strmlist)]

    c0 = interest - 0.5 * volatility ** 2
    c1 = volatility * math.sqrt(dt)

    # Configure the kernel
    # Similar to CUDA-C: cu_monte_carlo_pricer<<<gridsz, blksz, 0, stream>>>
    steplist = [cu_step[gridsz, blksz, strm]
               for gridsz, strm in zip(gridszlist, strmlist)]

    d_lastlist = [cuda.to_device(paths[s:e, 0], to=mm.get(stream=strm))
                  for (s, e), strm in zip(partitions, strmlist)]

    for j in range(1, paths.shape[1]):
        for prng, d_norm in zip(prnglist, d_normlist):
            prng.normal(d_norm, mean=0, sigma=1)

        d_pathslist = [cuda.to_device(paths[s:e, j], stream=strm,
                                      to=mm.get(stream=strm))
                       for (s, e), strm in zip(partitions, strmlist)]

        for step, args in zip(steplist, zip(d_lastlist, d_pathslist, d_normlist)):
            d_last, d_paths, d_norm = args
            step(d_last, d_paths, dt, c0, c1, d_norm)

        for d_paths, strm, (s, e) in zip(d_pathslist, strmlist, partitions):
            d_paths.copy_to_host(paths[s:e, j], stream=strm)
            mm.free(d_last, stream=strm)
        d_lastlist = d_pathslist

    for strm in strmlist:
        strm.synchronize()

if __name__ == '__main__':
    from driver import driver
    driver(monte_carlo_pricer, pinned=True)
