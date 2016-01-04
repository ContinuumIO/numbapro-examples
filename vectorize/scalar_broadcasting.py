'''
Demonstrate broadcasting when a scalar is provided as an argument to a 
vectorize function.

Please read NumPy Broadcasting documentation for details about broadcasting:
http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
'''

from __future__ import print_function
import numpy as np
from numba import vectorize, float32

@vectorize([float32(float32, float32, float32)], target='parallel')
def truncate(x, xmin, xmax):
    """ Truncate x[:] to [xmin, xmax] interval """
    if x < xmin:
        x = xmin
    elif x > xmax:
        x = xmax
    return x

def main():
    x = np.arange(100, dtype=np.float32)
    print('x = %s' % x)
    xmin = np.float32(20)  # as float32 type scalar
    xmax = np.float32(70)  # as float32 type scalar

    # The scalar arguments are broadcasted into an array.
    # This process creates arrays of zero strides.
    # The resulting array will contain exactly one element despite it
    # has a shape that matches that of `x`.
    out = truncate(x, xmin, xmax)

    print('out = %s' % out)

    # Check results
    assert np.all(out >= xmin)
    assert np.all(out <= xmax)

if __name__ == '__main__':
    main()
