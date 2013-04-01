# Mandelbrot

- `mandel_autojit.py`: implementation using `numba.autojit`.
- `mandel_cu.py`: implementation using numbapro CU API.
- `mandel_vectorize.py`: implementation using NumbaPro GPU vectorize.

## Running the examples

All scripts are runnable.  

`mandel_cu.py` --- takes an optional commandline argument for target selection.

```bash
python mandel_cu.py [cpu|gpu]
```
