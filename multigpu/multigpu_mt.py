from __future__ import print_function

import threading
from math import ceil

import numpy as np

from numba import cuda

print('System has %d CUDA devices' % len(cuda.list_devices()))

signature = 'void(int32[:], int32[:])'

def kernel(dst, src):
    '''A simple kernel that adds 1 to every item
    '''
    i = cuda.grid(1)
    if i >= dst.shape[0]:
        return
    dst[i] = src[i] + 1

# Numba compiler is not threadsafe
compiler_lock = threading.Lock()

def device_controller(cid):
    cuda.select_device(cid)                    # bind device to thread
    device = cuda.get_current_device()         # get current device

    # print some information about the CUDA card
    prefix = '[%s]' % device
    print(prefix, 'device_controller', cid, '| CC', device.COMPUTE_CAPABILITY)

    max_thread = device.MAX_THREADS_PER_BLOCK

    with compiler_lock:                        # lock the compiler
        # prepare function for this thread
        # the jitted CUDA kernel is loaded into the current context
        cuda_kernel = cuda.jit(signature)(kernel)

    # prepare data
    N = 12345
    data = np.arange(N, dtype=np.int32) * (cid + 1)
    orig = data.copy()

    # determine number of threads and blocks
    if N >= max_thread:
        ngrid = int(ceil(float(N) / max_thread))
        nthread = max_thread
    else:
        ngrid = 1
        nthread = N

    print(prefix, 'grid x thread = %d x %d' % (ngrid, nthread))

    # real CUDA work
    d_data = cuda.to_device(data)                   # transfer to device
    cuda_kernel[ngrid, nthread](d_data, d_data)     # compute inplace
    d_data.copy_to_host(data)                       # transfer to host

    # check result
    if not np.all(data == orig + 1):
        raise ValueError


def main():
    children = []
    for cid, dev in enumerate(cuda.list_devices()):
        t = threading.Thread(target=device_controller, args=(cid,))
        t.start()
        children.append(t)

    for t in children:
        t.join()

    print('ending gracefully')

if __name__ == '__main__':
    main()
