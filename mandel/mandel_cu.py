# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import
import sys
try:
    from numbapro import CU
except ImportError:
    raise Exception('''
As of NumbaPro 0.12.0, the experimental CU API was removed.  We will be
providing a better API for building parallel computation through composition
of high-level construct.
    ''')
from contextlib import closing
import numpy as np
from timeit import default_timer as timer
from pylab import imshow, jet, show, ion

def mandel(tid, min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    x = tid % width
    y = tid / width

    real = min_x + x * pixel_size_x
    imag = min_y + y * pixel_size_y

    c = complex(real, imag)
    z = 0.0j
    color = iters
    for i in range(iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            color = i
            break

    image[y, x] = color

def create_fractal(cu, min_x, max_x, min_y, max_y, image, iters):
    height, width = image.shape
    cu.enqueue(mandel,
               ntid=width * height,
               args=(min_x, max_x, min_y, max_y, image, iters))
    return image

def main():
    try:
        target = sys.argv[1]
    except:
        target = 'cpu'
    cu = CU(target)
    width = 500 * 10
    height = 750 * 10
    with closing(cu):
        image = np.zeros((width, height), dtype=np.uint8)
        d_image = cu.output(image)
        s = timer()
        create_fractal(cu, -2.0, 1.0, -1.0, 1.0, d_image, 20)
        cu.wait()
        e = timer()
    print('time: %f' % (e - s,))
#    print(image)
    imshow(image)
    show()


if __name__ == '__main__':
    main()
