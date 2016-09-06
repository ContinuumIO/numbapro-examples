'''
This example uses cuBLAS gemm routine to perform matrix-matrix multiplication.
Please refer to the documentation for details of how to use the gemm routine
  http://docs.continuum.io/accelerate/cublas#blas-level-2

Note: cuBLAS uses Fortran layout
'''
from __future__ import print_function

from timeit import default_timer as timer

import numpy as np

from accelerate.cuda.blas import Blas


N = 128     # no. of rows/cols


def gemm_v1():
    '''
    Note that all arrays are in Fortran order.
    '''
    print("Version 1".center(80, '='))
    # Prepare arrays for input
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N), order='F')
    B = np.array(np.arange(N) + 10, dtype=A.dtype, order='F')
    D = np.zeros_like(A, order='F')

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = Blas()

    start = timer()
    blas.gemm('N', 'N', N, N, N, 1.0, A, np.diag(B), 1.0, D)
    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


def gemm_v2():
    """
    Let GEMM transpose the input matrices so that they can be in C order,
    originally.  Note that the output matrix is still in Fortran array.
    The string arguments in gemm tells it to apply transformation on the input
    matrices.

    See argument description in:
        http://docs.continuum.io/accelerate/cublas#blas-level-2
    """
    print("Version 2".center(80, '='))
    # Prepare arrays for input
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N))
    B = np.array(np.arange(N) + 10, dtype=A.dtype)
    D = np.zeros_like(A, order='F')

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = Blas()

    start = timer()
    blas.gemm('T', 'T', N, N, N, 1.0, A, np.diag(B), 1.0, D)
    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


def main():
    gemm_v1()
    gemm_v2()


if __name__ == '__main__':
   main()

