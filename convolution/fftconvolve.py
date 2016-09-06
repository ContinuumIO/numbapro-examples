'''
This script executes 2D FFT convolution on images in grayscale.

Usage:

Run without argument will use builtin Lena image:

    python fftconvolve.py

Or, specify an image to use

    python fftconvolve.py myimage.jpg
    python fftconvolve.py myimage.png


= Getting The Requirements =

For Conda user, run the following to ensure the dependencies are fulfilled:

    conda install scipy matplotlib

You may need to install PIL from pip.

    conda install pip
    pip install PIL

'''
from __future__ import print_function

import sys
from timeit import default_timer as timer

import numpy as np
from scipy.signal import fftconvolve
from scipy import misc, ndimage
from matplotlib import pyplot as plt

from accelerate.cuda.fft import FFTPlan
from numba import cuda


@cuda.jit('void(complex64[:,:], complex64[:,:])')
def mult_inplace(img, resp):
    i, j = cuda.grid(2)
    if j < img.shape[0] and i < img.shape[1]:
        img[j, i] *= resp[j, i]


def best_grid_size(size, tpb):
    bpg = np.ceil(np.array(size, dtype=np.float) / tpb).astype(np.int).tolist()
    return tuple(bpg)


def main():
    # Build Filter
    laplacian_pts = '''
    -4 -1 0 -1 -4
    -1 2 3 2 -1
    0 3 4 3 0
    -1 2 3 2 -1
    -4 -1 0 -1 -4
    '''.split()

    laplacian = np.array(laplacian_pts, dtype=np.float32).reshape(5, 5)

    # Build Image
    try:
        filename = sys.argv[1]
        image = ndimage.imread(filename, flatten=True).astype(np.float32)
    except IndexError:
        image = misc.face(gray=True).astype(np.float32)

    print("Image size: %s" % (image.shape,))

    response = np.zeros_like(image)
    response[:5, :5] = laplacian

    # CPU
    ts = timer()
    cvimage_cpu = fftconvolve(image, laplacian, mode='same')
    te = timer()
    print('CPU: %.2fs' % (te - ts))

    # GPU
    threadperblock = 32, 8
    blockpergrid = best_grid_size(tuple(reversed(image.shape)), threadperblock)
    print('kernel config: %s x %s' % (blockpergrid, threadperblock))

    # Trigger initialization the cuFFT system.
    # This takes significant time for small dataset.
    # We should not be including the time wasted here
    FFTPlan(shape=image.shape, itype=np.complex64, otype=np.complex64)

    # Start GPU timer
    ts = timer()
    image_complex = image.astype(np.complex64)
    response_complex = response.astype(np.complex64)

    stream1 = cuda.stream()
    stream2 = cuda.stream()

    fftplan1 = FFTPlan(shape=image.shape, itype=np.complex64,
                       otype=np.complex64, stream=stream1)
    fftplan2 = FFTPlan(shape=image.shape, itype=np.complex64,
                       otype=np.complex64, stream=stream2)

    # pagelock memory
    with cuda.pinned(image_complex, response_complex):

        # We can overlap the transfer of response_complex with the forward FFT
        # on image_complex.
        d_image_complex = cuda.to_device(image_complex, stream=stream1)
        d_response_complex = cuda.to_device(response_complex, stream=stream2)

        fftplan1.forward(d_image_complex, out=d_image_complex)
        fftplan2.forward(d_response_complex, out=d_response_complex)

        stream2.synchronize()

        mult_inplace[blockpergrid, threadperblock, stream1](d_image_complex,
                                                            d_response_complex)
        fftplan1.inverse(d_image_complex, out=d_image_complex)

        # implicitly synchronizes the streams
        cvimage_gpu = d_image_complex.copy_to_host().real / np.prod(image.shape)

    te = timer()
    print('GPU: %.2fs' % (te - ts))

    # Plot the results
    plt.subplot(1, 2, 1)
    plt.title('CPU')
    plt.imshow(cvimage_cpu, cmap=plt.cm.gray)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('GPU')
    plt.imshow(cvimage_gpu, cmap=plt.cm.gray)
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()

