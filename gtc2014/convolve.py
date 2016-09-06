"""
Convolve
"""
from __future__ import print_function
import sys
import numpy as np
from scipy.signal import fftconvolve
from scipy import misc, ndimage
from accelerate.cuda.fft import FFTPlan, fft_inplace, ifft_inplace
from numba import cuda, vectorize
from timeit import default_timer as timer

@vectorize(['complex64(complex64, complex64)'], target='cuda')
def vmult(a, b):
    """Element complex64 multiplication
    """
    return a * b


def task1(d_image_complex, d_response_complex):
    ### Task1 ###
    # Implement a inplace CUDA FFT convolution
    # Pseduocode:
    #   freq_imag = fft(image)
    #   freq_resp = fft(response)
    #   freq_out = fftimag * fftresp
    #   output = ifft(freq_out)
    #
    # Use the cuFFT functions:
    #   - fft_inplace(ary)
    #   - ifft_inplace(ary)
    #
    # Call `vmult` which is our elementwise complex multiplication.
    # Do a inplace operation on `d_image_complex`.
    # Hints:
    #   - keyword argument 'out' specify the output array
    #   - length of d_image_complex and d_response_complex has the same length.


    fft_inplace(d_image_complex)
    fft_inplace(d_response_complex)

    vmult(d_image_complex, d_response_complex, out=d_image_complex)

    ifft_inplace(d_image_complex)

    # At this point, we have applied the filter onto d_image_complex
    return  # Does not return anything


# ----------------------------------------------------------------------------
# Details of the program.  Not necessary for the exercise.

def convolve():
    # Build Filter
    laplacian_pts = '''
    -4 -1 0 -1 -4
    -1 2 3 2 -1
    0 3 4 3 0
    -1 2 3 2 -1
    -4 -1 0 -1 -4
    '''.split()

    laplacian = np.array(laplacian_pts, dtype=np.float32).reshape(5, 5)

    image = get_image()

    print("Image size: %s" % (image.shape,))

    response = np.zeros_like(image)
    response[:5, :5] = laplacian

    # CPU
    # Use SciPy to perform the FFT convolution
    ts = timer()
    cvimage_cpu = fftconvolve(image, laplacian, mode='same')
    te = timer()
    print('CPU: %.2fs' % (te - ts))

    # GPU
    threadperblock = 32, 8
    blockpergrid = best_grid_size(tuple(reversed(image.shape)), threadperblock)
    print('kernel config: %s x %s' % (blockpergrid, threadperblock))

    # Initialize the cuFFT system.
    FFTPlan(shape=image.shape, itype=np.complex64, otype=np.complex64)

    # Start GPU timer
    ts = timer()
    image_complex = image.astype(np.complex64)
    response_complex = response.astype(np.complex64)

    d_image_complex = cuda.to_device(image_complex)
    d_response_complex = cuda.to_device(response_complex)

    task1(d_image_complex, d_response_complex)

    cvimage_gpu = d_image_complex.copy_to_host().real / np.prod(image.shape)

    te = timer()
    print('GPU: %.2fs' % (te - ts))

    return cvimage_cpu, cvimage_gpu


def best_grid_size(size, tpb):
    bpg = np.ceil(np.array(size, dtype=np.float) / tpb).astype(np.int).tolist()
    return tuple(bpg)

def get_image():
    # Build Image
    try:
        filename = sys.argv[1]
        image = ndimage.imread(filename, flatten=True).astype(np.float32)
    except IndexError:
        image = misc.face(gray=True).astype(np.float32)
    return image

def main():
    cvimage_cpu, cvimage_gpu = convolve()

    # Plot the results
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        print("To render the images")
        print("Do `conda install matplotlib` on the terminal")
        print("Then, rerun")
    else:
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
