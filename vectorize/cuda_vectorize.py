'''
Demonstrate the vectorize API with automatical memory transfer and
manual memory transfer.
'''
from __future__ import print_function
from timeit import default_timer as timer
from numba import vectorize, float64, cuda
import numpy


@vectorize([float64(float64, float64)], target='cuda')
def vector_mul(a, b):
    return  a * b

a = numpy.random.rand(10000000)
b = numpy.random.rand(10000000)

# Let NumbaPro automatically convert host memory to device memory
ts = timer()
for i in range(10):
    result = vector_mul(a, b)
te = timer()

print('auto', te - ts)


# Manual conversion between host and device memory
ts = timer()
for i in range(10):
    # copy host memory to device
    da = cuda.to_device(a)
    db = cuda.to_device(b)
    # execute kernel

    # When device array is used as argument, the output will be a device
    # array instead of a host array.
    dresult = vector_mul(da, db)

    # copy device memory to host
    result = dresult.copy_to_host()

# make sure device memory is freed for comparable benchmark
del da
del db
del dresult
te = timer()

print('manual', te - ts)

# The reason that the manual version is faster is due to the timing for which
# the device memory is freed.
