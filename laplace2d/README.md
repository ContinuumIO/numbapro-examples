# Laplace 2D 

Implements jacobian relaxation on 4096x4096 matrices.

- laplace2d.py: naive implementation; very slow.
- laplace2d-numba.py: numba implementation.
- laplace2d-numba-gpu.py: naive numba cuda implementation.
- laplace2d-numba-gpu-smem.py: shared memory version of cuda implementation.
- laplace2d-numba-gpu-improve.py: shared memory + inline reduction on cuda.

Each script is runnable.
