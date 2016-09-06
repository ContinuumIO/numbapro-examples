'''
Demonstrate the significant performance difference between transferring
regular host memory and pinned (pagelocked) host memory.
'''
from __future__ import print_function

from timeit import default_timer as timer

import numpy as np

from numba import vectorize, float32, cuda

src = np.arange(10 ** 7, dtype=np.float32)
dst = np.empty_like(src)


@vectorize([float32(float32)], target='cuda')
def copy_kernel(src):
    return src

# Regular memory transfer

ts = timer()
d_src = cuda.to_device(src)
d_dst = cuda.device_array_like(dst)

copy_kernel(d_src, out=d_dst)

d_dst.copy_to_host(dst)
te = timer()

print('regular', te - ts)

del d_src, d_dst

assert np.allclose(dst, src)

# Pinned (pagelocked) memory transfer

with cuda.pinned(src, dst):
    ts = timer()
    stream = cuda.stream()  # use stream to trigger async memory transfer
    d_src = cuda.to_device(src, stream=stream)
    d_dst = cuda.device_array_like(dst, stream=stream)

    copy_kernel(d_src, out=d_dst, stream=stream)

    d_dst.copy_to_host(dst, stream=stream)
    stream.synchronize()
    te = timer()
    print('pinned', te - ts)

assert np.allclose(dst, src)
