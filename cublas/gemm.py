'''
This example uses cuBLAS gemm routine to perform matrix-matrix multiplication.
Please refer to the documentation for details of how to use the gemm routine
  http://docs.continuum.io/numbapro/cudalib.html#blas-level-2
  
Note: cuBLAS uses Fortran layout
'''

import numbapro.cudalib.cublas as cublas
import numpy as np
from timeit import default_timer as timer

def main():
    # Prepare arrays for input
    N = 128
    A = np.array(np.arange(N ** 2, dtype=np.float32).reshape(N, N), order='F')
    B = np.array(np.arange(N) + 10, dtype=A.dtype, order='F')
    D = np.zeros_like(A)

    # NumPy
    start = timer()
    E = np.dot(A, np.diag(B))
    numpy_time = timer() - start
    print("Numpy took %f seconds" % numpy_time)

    # cuBLAS
    blas = cublas.api.Blas()
    
    start = timer()
    blas.gemm('N', 'N', N, N, N, 1.0, A, np.diag(B), 1.0, D)

    cuda_time = timer() - start

    print("CUBLAS took %f seconds" % cuda_time)
    diff = np.abs(D - E)
    print("Maximum error %f" % np.max(diff))


if __name__ == '__main__':
   main()

