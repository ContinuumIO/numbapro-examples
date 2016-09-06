from __future__ import print_function

import numpy as np

from numba import cuda, int32


@cuda.jit('int32(int32, int32)', device=True, inline=True)
def mymax(a, b):
    if a > b:
        return a
    else:
        return b


@cuda.jit('void(int32[:], int32[:])')
def max_kernel(a, b):
    "Simple implementation of reduction kernel"
    # Allocate static shared memory of 256.
    # This limits the maximum block size to 256.
    sa = cuda.shared.array(shape=(256,), dtype=int32)
    tx = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bw = cuda.blockDim.x
    i = tx + bx * bw
    if i < a.shape[0]:
        sa[tx] = a[i]
        if tx == 0:
            # Uses the first thread of each block to perform the actual
            # reduction
            m = sa[tx]
            cuda.syncthreads()
            for j in range(1, bw):
                m = mymax(m, sa[j])
            b[bx] = m

blkct = 4
n = 20

a = np.random.randint(0, n, n).astype(np.int32)
b = np.empty(blkct, dtype=np.int32)

d_a = cuda.to_device(a)
d_b = cuda.to_device(b, copy=False)

griddim = blkct, 1
blockdim = a.size//blkct, 1

max_kernel[griddim, blockdim](d_a, d_b)

d_b.to_host()

print('a =', a)
print('b =', b)
print('np.max(b) =', np.max(b))
assert np.max(b) == np.max(a)

