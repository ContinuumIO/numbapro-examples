"""
Demonstrating CUDA JIT integration
"""
from __future__ import print_function
from numba import cuda
import numpy
import os

# Declare function to link to
bar = cuda.declare_device('bar', 'int32(int32, int32)')

# Get path to precompiled library
curdir = os.path.join(os.path.dirname(__file__))
link = os.path.join(curdir, 'jitlink.o')
print("Linking: %s", link)

# Code that uses CUDA JIT
@cuda.jit('void(int32[:], int32[:])', link=[link])
def foo(inp, out):
	i = cuda.grid(1)
	out[i] = bar(inp[i], 2)

print(foo.ptx)


n = 5
inp = numpy.arange(n, dtype='int32')
out = numpy.zeros_like(inp)
foo[1, out.size](inp, out)

print("inp =", inp)
print("out =", out)

