'''
This example uses cuBLAS gemm routine to perform matrix-matrix multiplication.
Please refer to the documentation for details of how to use the gemm routine
  http://docs.continuum.io/numbapro/cudalib.html#blas-level-2
  
Note: cuBLAS uses Fortran layout
'''

from numbapro import jit, cuda
from numba import float32
import numbapro.cudalib.cublas as cublas
import numpy as np
from timeit import default_timer as timer
 
def generate_input(n):
    import numpy as np
    A = np.array(np.arange(n ** 2, dtype=np.float32).reshape(n,n), order='F')
    B = np.array(np.arange(n) + 10, dtype=A.dtype, order='F')
    return A, B
 
def main():
   
    N = 128
 
    A, B = generate_input(N)
    D = np.zeros_like(A)
    E = np.empty(A.shape)

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
    diff = np.abs(D-E)
    print("Maximum error %f" % np.max(diff))


 
if __name__ == '__main__':
   main()

