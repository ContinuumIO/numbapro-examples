'''Example: sum each row using guvectorize

See Numpy documentation for detail about gufunc:
    http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
'''
from __future__ import print_function

import sys

import numpy as np

from numba import guvectorize, cuda

if sys.version_info[0] == 2:
    range = xrange

# Controls whether to manually handle CUDA memory allocation or not.
MANAGE_CUDA_MEMORY = True

#    function type:
#        - has no void return type
#        - array argument is one dimenion fewer than the source array
#        - scalar output is passed as a 1-element array.
#
#    signature: (n)->()
#        - the function takes an array of n-element and output a scalar.

@guvectorize(['void(int32[:], int32[:])'], '(n)->()', target='cuda')
def sum_row(inp, out):
    tmp = 0.
    for i in range(inp.shape[0]):
        tmp += inp[i]
    out[0] = tmp

# inp is (10000, 3)
# out is (10000)
# The outter (leftmost) dimension must match or numpy broadcasting is performed.
# But, broadcasting on CUDA arrays is not supported.

inp = np.arange(30000, dtype=np.int32).reshape(10000, 3)


if MANAGE_CUDA_MEMORY:
    # invoke on CUDA with manually managed memory
    out = np.empty(10000, dtype=inp.dtype)

    dev_inp = cuda.to_device(inp)             # alloc and copy input data
    dev_out = cuda.to_device(out, copy=False) # alloc only

    sum_row(dev_inp, out=dev_out)             # invoke the gufunc

    dev_out.copy_to_host(out)                 # retrieve the result
else:
    # Manually managing the CUDA allocation is optional, but recommended
    # for maximum performance.
    out = sum_row(inp)

# verify result
goal = np.empty_like(out)
for i in range(inp.shape[0]):
    assert out[i] == inp[i].sum()

# print out
print('input'.center(80, '-'))
print(inp)
print('output'.center(80, '-'))
print(out)

