# Demonstrate Multi GPU support

Only one CUDA context can be associated with a thread at any moment.
In CUDA-C, users can push and pop context to switch between context of different devices.
In CUDA Python, push/pop contexts is not supported yet.
On the another hand, we support a multithreaded model in doing so since using threads in Python is easy and portable.

## Beware

While the CUDA calls are threadsafe, the Numba compiler is not.
Users must lock whenever calling any compiling functions (e.g. jit).


## Files

- `multigpu_mt.py`
