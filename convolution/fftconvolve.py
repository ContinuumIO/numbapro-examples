import sys
import numpy as np
from scipy.signal import fftconvolve
from scipy import misc, ndimage
from matplotlib import pyplot as plt
from numbapro.cudalib import cufft
from numbapro import cuda
from timeit import default_timer as timer

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
        image = misc.lena().astype(np.float32)

    print("Image size: %s" % (image.shape,))

    response = np.zeros_like(image)
    response[:5, :5] = laplacian

    # CPU
    ts = timer()
    cvimage_cpu = fftconvolve(image, laplacian, mode='same')
    te = timer()
    print('CPU: %.2fs' % (te - ts))

    # GPU
    threadperblock = 32, 16
    blockpergrid = best_grid_size(tuple(reversed(image.shape)), threadperblock)
    print('kernel config: %s x %s' % (blockpergrid, threadperblock))

    ts = timer()
    image_complex = image.astype(np.complex64)
    response_complex = response.astype(np.complex64)

    stream = cuda.stream()

    d_image_complex = cuda.to_device(image_complex, stream=stream)
    d_response_complex = cuda.to_device(response_complex, stream=stream)

    cufft.fft_inplace(d_image_complex, stream=stream)
    cufft.fft_inplace(d_response_complex, stream=stream)
    mult_inplace[blockpergrid, threadperblock, stream](d_image_complex, d_response_complex)
    cufft.ifft_inplace(d_image_complex, stream=stream)
    cvimage_gpu = d_image_complex.copy_to_host().real / np.prod(image.shape)

    te = timer()
    print('CPU: %.2fs' % (te - ts))

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

