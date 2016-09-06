'''
Demonstrate cuda.event

Python measured time for the kernel should be much shorter than the cuda.event
time because kernel calls are aynchronous---it returns before the work in
completed.
'''

from __future__ import print_function

from timeit import default_timer as timer

import numpy as np

from numba import cuda


@cuda.jit('void(float32[:], float32[:])')
def cu_copy_array(dst, src):
    i = cuda.grid(1)
    dst[i] = src[i]


BLOCKCOUNT = 25000
BLOCKSIZE = 256

aryA = np.arange(BLOCKSIZE * BLOCKCOUNT, dtype=np.float32)

print('data size: %.1fMB' % (aryA.size * aryA.dtype.itemsize / (2**20)))

evt_total_begin = cuda.event()
evt_total_end = cuda.event()

evt_kernel_begin = cuda.event()
evt_kernel_end = cuda.event()

t_total_begin = timer()
evt_total_begin.record()

# explicity tranfer memory
d_aryA = cuda.to_device(aryA)
d_aryB = cuda.device_array_like(aryA)

evt_kernel_begin.record()

t_kernel_begin = timer()
cu_copy_array[BLOCKCOUNT, BLOCKSIZE](d_aryB, d_aryA)
t_kernel_end = timer()

evt_kernel_end.record()

aryB = d_aryB.copy_to_host()

evt_total_end.record()

evt_total_end.synchronize()
t_total_end = timer()

assert np.all(aryA == aryB)

print('CUDA EVENT TIMING'.center(80, '='))
print('total time: %fms' % evt_total_begin.elapsed_time(evt_total_end))
print('kernel time: %fms' % evt_kernel_begin.elapsed_time(evt_kernel_end))
print('PYTHON TIMING'.center(80, '='))
print('total time: %fms' % (float(t_total_end - t_total_begin) * 1000))
print('kernel time: %fms' % (float(t_kernel_end - t_kernel_begin) * 1000))


