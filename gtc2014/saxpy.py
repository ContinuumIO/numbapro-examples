"""
SAXPY
-----

Compute `a * x + y`

Where
	`a` is a scalar
	`x` and `y` are vectors

Prefix 'S' indicates single-precision float32 operations
"""
from __future__ import print_function
import sys
import numpy
from numba import cuda, vectorize, float32, void

# GPU code
# ---------

@cuda.jit(void(float32, float32[:], float32[:], float32[:]))
def saxpy(a, x, y, out):
	# Short for cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
	i = cuda.grid(1)
	# Map i to array elements
	if i >= out.size:
		# Out of range?
		return
	# Do actual work
	out[i] = a * x[i] + y[i]

"""
Vectorize turns a scalar function into a
elementwise operation over the input arrays.
"""

@vectorize([float32(float32, float32, float32)],
		   target='cuda')
def vec_saxpy(a, x, y):
	### Task 1 ###
	# Complete the vectorize version
	# Hint: this is a scalar function of
	# 		float32(float32 a, float32 x, float32 y)
	return a * x + y


# CPU code
# ---------

NUM_BLOCKS = 1
NUM_THREADS = 32
NELEM = NUM_BLOCKS * NUM_THREADS


def task1():
	a = numpy.float32(2.)				# Force value to be float32

	# Generate numbers 0..(NELEM - 1)
	x = numpy.arange(NELEM, dtype='float32')
	y = numpy.arange(NELEM, dtype='float32')
	out = numpy.empty_like(x)

	griddim = NUM_BLOCKS
	blockdim = NUM_THREADS
	saxpy[griddim, blockdim](a, x, y, out)
	print("out =", out)

	vecout = vec_saxpy(a, x, y)

	print("vecout =", vecout)

	# Check output
	if not (out == vecout).all():
		print("Incorrect result")
	else:
		print("Correct result")


def task2():
	a = numpy.float32(2.)				# Force value to be float32
	x = numpy.arange(NELEM, dtype='float32')
	y = numpy.arange(NELEM, dtype='float32')

	### Task2 ###
	# a) Complete the memory transfer for x -> dx, y -> dy
	# b) Allocate device memory for dout
	# c) Transfer for out <- dout
	dx = cuda.to_device(x)
	dy = cuda.to_device(y)
	dout = cuda.device_array_like(x)

	griddim = NUM_BLOCKS
	blockdim = NUM_THREADS
	saxpy[griddim, blockdim](a, dx, dy, dout)

	out = dout.copy_to_host()
	print("out =", out)

	if numpy.allclose(a * x + y, out):
		print("Correct result")
	else:
		print("Incorrect result")


# ----------------------------------------------------------------------------
# All exercise code is above

HELPER = """
Lab 1

This exercise contains two tasks.

Usage:

	python %(this)s --task1

		Execute task1 codepath

	python %(this)s --task2

		Execute task2 codepath
"""

def main():
	if '--task1' in sys.argv[1:]:
		task1()
	elif '--task2' in sys.argv[1:]:
		task2()
	else:
		print(HELPER % {'this': sys.argv[0]})

if __name__ == '__main__':
	main()



