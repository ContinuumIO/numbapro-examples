"""
# CUDA Matrix Multiplication with Shared Memory


This is a sequel to the [CUDA Matrix Multiplication](matrixmult_intro.html)
example.

This document introduces the CUDA shared memory.  We will demonstrate its
usage for accelerating the matrix multiplication code developed preivously.
"""

from __future__ import absolute_import, print_function, division

# builtin packages
import sys
import datetime
from timeit import default_timer as timer
# extra packages
import numpy as np
import numba
from numba import cuda, float32

"""
**Version information:**
"""

print("This file is generated on:", datetime.datetime.now())
print("python: {0}.{1}".format(*sys.version_info[:2]))
print("numpy:", np.__version__)
print("numba:", numba.__version__)
print("CUDA GPU:", cuda.gpus[0].name)

"""
## A Naive GPU Version

We will reuse the matrix multiplication code from the previous example.
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
## Optimize with Shared Memory and Block Algorithm

A faster way to implement the matrix multiplication is to use a block
algorithm. The CUDA thread hierarchy fits naturally to this.  We can map CUDA
blocks to matrix blocks and map CUDA threads to elements in each block.

### How is this faster?

To understand, we need to understand a little bit about the CUDA **memory
hierarchy**. Unlike CPU, CUDA exposes a *shared memory* unit that is private to
each block and acts like a manual cache.  Data transfer from the shared
memory is a lot faster than from the *global memory*, which is accessible from
the GPU and CPU.  The ``cuda.to_device()`` puts data to the *global memory*.

The block algorithm loads each matrix block into the shared memory before
computing the product for the block.  This allows all the threads computing
on the current matrix block to reuse memory loaded into the shared memory
(our manual cache) instead of from the slower global memory.

### Synchronization

In each CUDA block, threads are cooperatively loading data into shared
memory.  These threads are running concurrently and they may not be executing
the same instruction.  We need a barrier ``cuda.syncthreads()`` to ensure
all the threads have executed up to a certain point.  We need one before the
data preload to sure the shared memory is not modified while some threads are
still using it.  We need one after the data preload to ensure all the threads
have completed the preloading.

**Important Note**

The barrier ``cuda.syncthreads()`` blocks **all** threads in the current block
until all of them have reached the same location.  The behavior is
**undefined** if some threads have returned.  Therefore, we need to keep all
the threads in the block alive if some of them may execute a barrier.
"""

block_per_grid = 10
thread_per_block = 16


@cuda.jit
def gpu_blocked_matrix_mult(matA, matB, matC):
    # Define shared array
    smA = cuda.shared.array(shape=(thread_per_block, thread_per_block),
                            dtype=float32)
    smB = cuda.shared.array(shape=(thread_per_block, thread_per_block),
                            dtype=float32)

    # Get thread IDs
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bw = cuda.blockDim.x
    bh = cuda.blockDim.y

    # Get global ID
    x = tx + bx * bw
    y = ty + by * bh

    # Get bounds
    m, n = matC.shape

    # Bound check
    in_bound = x < m and y < n

    # Computation starts here
    acc = 0

    # For each block
    for i in range(block_per_grid):
        # Wait for all threads to reach this point
        cuda.syncthreads()

        if in_bound:
            # Cooperatively load from global memory into faster shared memory
            smA[tx, ty] = matA[x, ty + i * thread_per_block]
            smB[tx, ty] = matB[tx + i * thread_per_block, y]

        # Wait for all threads to reach this point
        cuda.syncthreads()

        if in_bound:
            # Compute using data in shared memory
            for j in range(thread_per_block):
                acc += smA[tx, j] * smB[j, ty]

    if in_bound:
        # Store result
        matC[x, y] = acc


"""
Check correctness
"""

mat_dim_large = [block_per_grid * thread_per_block] * 2

matA = np.random.random(mat_dim_large).astype(np.float32)
matB = np.random.random(mat_dim_large).astype(np.float32)

gpu_result = np.zeros_like(matA)

griddim = block_per_grid, block_per_grid
blockdim = thread_per_block, thread_per_block

gpu_blocked_matrix_mult[griddim, blockdim](matA, matB, gpu_result)

npy_result = np.dot(matA, matB)

assert np.allclose(npy_result, gpu_result)

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
Generate timing
"""

res_naive = np.empty_like(matA)
res_blocked = np.empty_like(matA)

gpu1_time = time_took(
    lambda: gpu_matrix_mult[griddim, blockdim](matA, matB, res_naive)
)
gpu2_time = time_took(
    lambda: gpu_blocked_matrix_mult[griddim, blockdim](matA, matB, res_blocked)
)

# result matches?
assert np.allclose(res_naive, res_blocked)
# faster?
assert gpu2_time < gpu1_time

fmt = "{0:>30s}: {1:.4f} seconds"
print(fmt.format("naive version", gpu1_time))
print(fmt.format("blocked+sharedmemory version", gpu2_time))
print("Speedup: {0:.1f}x".format(gpu1_time / gpu2_time))
