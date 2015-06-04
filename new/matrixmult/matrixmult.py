"""
# CUDA Matrix Multiplication

This document introduces CUDA programming with a simple GPU square matrix
multiplication.  The implementation used here is for demonstrating the CUDA
parallel programming and the code is not optimized for high performance.
"""

from __future__ import absolute_import, print_function, division

# builtin packages
import sys
import datetime
from timeit import default_timer as timer
# extra packages
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
## A CPU Version

This implements a square matrix multiplication using a naive algorithm.  We
compile it with Numba for speed.
"""


@jit
def cpu_matrix_mult(matA, matB, matC):
    m, n = matC.shape
    k = matB.shape[0]
    for x in range(m):
        for y in range(n):
            matC[x, y] = 0
            for i in range(k):
                matC[x, y] += matA[x, i] * matB[i, y]


"""
## Testing the CPU Code
"""

"""
Create small matrices for testing
"""

mat_dim_small = 4, 4
matA_small = np.random.random(mat_dim_small).astype(np.float32)
matB_small = np.random.random(mat_dim_small).astype(np.float32)
cpu_result = np.zeros_like(matA_small)

"""
Execute
"""
cpu_matrix_mult(matA_small, matB_small, cpu_result)

"""
Check results
"""

print("CPU result")
print(cpu_result)
assert np.allclose(np.dot(matA_small, matB_small), cpu_result)

"""
## A CUDA GPU Version

This implements a CUDA GPU version of the matrix multiply.
We are using ``@cuda.jit`` to decorate the implementation to compile it into
a *CUDA kernel*.  When the kernel function is launched, every thread
will execute the same code.  To tell which thread the execution is in,
CUDA provides a set of special registers that are accessible with
``cuda.threadIdx``, ``cuda.blockIdx`` and ``cuda.blockDim``.  These registers
are 2D or 3D vectors of the thread ID, block ID and block dimension,
representively.

CUDA defines a **thread hierarchy**.  A kernel launch creates a **grid** of
**blocks**.  Each block contains **threads**.

A common pattern is to compute the global thread ID, as oppose to
using the nested thread and block IDs.  In this example, the kernel is launched
with a 2D grid and 2D block that the combined dimension matches the shape of
the matrices.  Therefore, the flattened global thread ID maps directly to the
indices of each element in the matrix.

For cases where the matrix shape is not multiple of the block dimension,
a common practice is to launch more threads than there are elements.  The extra
thread will have nothing to do.  It is important to check for these threads
to avoid invalid memory reads and writes.
"""


@cuda.jit
def gpu_matrix_mult(matA, matB, matC):
    # Read special register for thread ID
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # Get global thread ID
    x = tx + bx * bw
    y = ty + by * bh

    # Get bounds
    m, n = matC.shape
    k = matB.shape[0]

    # Check for out-of-bound
    if x >= m or y >= n:
        # This is an extra thread.  Exit.
        return

    # The actual computation per output element
    res = 0
    for i in range(k):
        res += matA[x, i] * matB[i, y]
    # Store the result
    matC[x, y] = res


"""
## Testing the CUDA Code

Decide of CUDA grid/block dimensions.
"""

block_per_grid = 60
thread_per_block = 16

griddim = block_per_grid, block_per_grid
blockdim = thread_per_block, thread_per_block

"""
Create matrices using base on the grid/block dimensions.
"""

mat_dim_large = [block_per_grid * thread_per_block] * 2

matA = np.random.random(mat_dim_large).astype(np.float32)
matB = np.random.random(mat_dim_large).astype(np.float32)

gpu_result = np.zeros_like(matA)

"""
Launch kernel

The square bracket ``[]`` is overloaded to configure the launch for the grid
and block dimensions.
"""

gpu_matrix_mult[griddim, blockdim](matA, matB, gpu_result)

"""
Check result
"""

npy_result = np.dot(matA, matB)
assert np.allclose(npy_result, gpu_result)
print("L1 norm", np.linalg.norm(gpu_result - npy_result, ord=1))

"""
## Optimizing Memory Transfers

By default, numba automatically transfer numpy array memory between
the CPU and GPU.  This is convenient but may lead to redundant memory
transfers.  Numba will always transfer numpy array back to the CPU.
User can control the memory transfer explicit to optimize the process.

To copy to the GPU device from the CPU host
"""

device_matA = cuda.to_device(matA)
device_matB = cuda.to_device(matB)
print(device_matA)
print(device_matB)

"""
To allocate GPU memory directly.  (It is similar to ``numpy.empty_like``.)
"""

device_matC = cuda.device_array_like(matA)
print(device_matC)

"""
Launch
"""

gpu_matrix_mult[griddim, blockdim](device_matA, device_matB, device_matC)

"""
Copy GPU device memory back to CPU host
"""

gpu_result = device_matC.copy_to_host()
print(gpu_result)

"""
## Comparing Speed
"""

"""
A function for timing function execution
"""


def time_took(functor):
    ts = timer()
    functor()
    te = timer()
    return te - ts


"""
A function for that uses ``gpu_matrix_mult()`` with manual memory transfer
"""


def gpu_manual_memory(matA, matB):
    device_matC = cuda.device_array_like(matA)
    device_matA = cuda.to_device(matA)
    device_matB = cuda.to_device(matB)
    gpu_matrix_mult[griddim, blockdim](device_matA, device_matB, device_matC)
    device_matC.copy_to_host()


"""
Generate timing
"""

res = np.empty_like(matA)
cpu_time = time_took(lambda: cpu_matrix_mult(matA, matB, res))
gpu1_time = time_took(lambda: gpu_matrix_mult[griddim, blockdim](matA, matB,
                                                                 res))
gpu2_time = time_took(lambda: gpu_manual_memory(matA, matB))

assert gpu2_time < gpu1_time < cpu_time

fmt = "{0:>40s}: {1:.2f} seconds"
print(fmt.format("numba cpu matrix mult", cpu_time))
print(fmt.format("numba gpu matrix mult (auto transfer)", gpu1_time))
print(fmt.format("numba gpu matrix mult (manual transfer)", gpu2_time))
