from __future__ import print_function

import sys
import datetime
from timeit import default_timer as timer
import numpy as np

import numba
from numba import cuda, jit

"""
**Version information:**
"""

print("This file is generated on:", datetime.datetime.now())
print("python: {0}.{1}".format(*sys.version_info[:2]))
print("numpy:", np.__version__)
print("numba:", numba.__version__)
print("CUDA GPU:", cuda.gpus[0].name)

"""
## NumPy CPU Jacobi Relaxation
"""


def numpy_jacobi_core(A, Anew):
    error = 0.0
    n = A.shape[0]
    m = A.shape[1]

    for j in range(1, n - 1):
        for i in range(1, m - 1):
            Anew[j, i] = 0.25 * (A[j, i + 1] + A[j, i - 1] \
                                 + A[j - 1, i] + A[j + 1, i])
            error = max(error, abs(Anew[j, i] - A[j, i]))
    return error


"""
The CPU driver
"""


def cpu_driver(corefn, NN, NM):
    A = np.zeros((NN, NM), dtype=np.float32)
    Anew = np.zeros((NN, NM), dtype=np.float32)

    n = NN
    m = NM
    iter_max = 1000

    tol = 1.0e-6
    error = 1.0

    for j in range(n):
        A[j, 0] = 1.0
        Anew[j, 0] = 1.0

    print("Jacobi relaxation Calculation: {0} x {1} mesh".format(n, m))

    ts = timer()
    iter = 0

    while error > tol and iter < iter_max:
        error = corefn(A, Anew)

        # swap A and Anew
        tmp = A
        A = Anew
        Anew = tmp

        if iter % 100 == 0:
            print("{0:5}, {1:.6f} (elapsed: {2} s)".format(iter, error,
                                                           timer() - ts))

        iter += 1

    runtime = timer() - ts
    print(" total: {0} s".format(runtime))
    return runtime


"""
Test NumPy Version
"""

numpy_runtime = cpu_driver(numpy_jacobi_core, 64, 64)

"""
## Numba JIT version

Compile
"""

numba_jacobi_core = jit(numpy_jacobi_core)

"""
Test Numba Version
"""
numba_runtime = cpu_driver(numba_jacobi_core, 64, 64)

"""
Speedup: Numba vs NumPy
"""

print("Speedup {0}x".format(numpy_runtime / numba_runtime))

"""
## CUDA Version
"""

tpb = 16


@cuda.jit
def cuda_jacobi_core(A, Anew, error):
    err_sm = cuda.shared.array((tpb, tpb), dtype=float32)

    ty = cuda.threadIdx.x
    tx = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y

    n = A.shape[0]
    m = A.shape[1]

    i, j = cuda.grid(2)

    err_sm[ty, tx] = 0
    if j >= 1 and j < n - 1 and i >= 1 and i < m - 1:
        Anew[j, i] = 0.25 * (A[j, i + 1] + A[j, i - 1] \
                             + A[j - 1, i] + A[j + 1, i])
        err_sm[ty, tx] = Anew[j, i] - A[j, i]

    cuda.syncthreads()

    # max-reduce err_sm vertically
    t = tpb // 2
    while t > 0:
        if ty < t:
            err_sm[ty, tx] = max(err_sm[ty, tx], err_sm[ty + t, tx])
        t //= 2
        cuda.syncthreads()

    # max-reduce err_sm horizontally
    t = tpb // 2
    while t > 0:
        if tx < t and ty == 0:
            err_sm[ty, tx] = max(err_sm[ty, tx], err_sm[ty, tx + t])
        t //= 2
        cuda.syncthreads()

    if tx == 0 and ty == 0:
        error[by, bx] = err_sm[0, 0]


"""
A GPU driver
"""


def gpu_driver(gpucorefn, NN, NM):
    A = np.zeros((NN, NM), dtype=np.float32)
    Anew = np.zeros((NN, NM), dtype=np.float32)

    n = NN
    m = NM
    iter_max = 1000

    tol = 1.0e-6
    error = 1.0

    for j in range(n):
        A[j, 0] = 1.0
        Anew[j, 0] = 1.0

    print("Jacobi relaxation Calculation: {0} x {1} mesh".format(n, m))

    ts = timer()
    iter = 0

    blockdim = (tpb, tpb)
    griddim = (NN // blockdim[0], NM // blockdim[1])

    error_grid = np.zeros(griddim, dtype=np.float32)

    stream = cuda.stream()

    dA = cuda.to_device(A, stream)  # to device and don't come back
    dAnew = cuda.to_device(Anew, stream)  # to device and don't come back
    derror_grid = cuda.to_device(error_grid, stream)

    while error > tol and iter < iter_max:
        gpucorefn[griddim, blockdim, stream](dA, dAnew, derror_grid)

        derror_grid.to_host(stream)

        # error_grid is available on host
        stream.synchronize()

        error = np.abs(error_grid).max()

        # swap dA and dAnew
        dA, dAnew = dAnew, dA

        if iter % 100 == 0:
            print("{0:5}, {1:.6f} (elapsed: {2} s)".format(iter, error,
                                                           timer() - ts))

        iter += 1

    runtime = timer() - ts
    print(" total: {0} s".format(runtime))
    return runtime


"""
Test CUDA Version
"""

gpu_driver(cuda_jacobi_core, 64, 64)

"""
Testing on Large Mesh
"""

NN = NM = 2048
print("Numba CPU:")
numba_runtime = cpu_driver(numba_jacobi_core, NN, NM)
print("Numba CUDA:")
cuda_runtime = gpu_driver(cuda_jacobi_core, NN, NM)

print("CUDA speedup: {0}x".format(numba_runtime / cuda_runtime))
