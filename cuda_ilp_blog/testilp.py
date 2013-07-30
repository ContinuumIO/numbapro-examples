from timeit import default_timer as timer
import math
import numpy as np
import pylab
from numbapro import cuda

@cuda.jit('float32(float32, float32)', device=True)
def core(a, b):
    return (a + b) #** 2 / 2

@cuda.jit('void(float32[:], float32[:], float32[:])')
def vec_add(a, b, c):
    i = cuda.grid(1)
    c[i] = core(a[i], b[i])

@cuda.jit('void(float32[:], float32[:], float32[:])')
def vec_add_ilp_x2(a, b, c):
    # read
    i = cuda.grid(1)
    ai = a[i]
    bi = b[i]

    bw = cuda.blockDim.x
    gw = cuda.gridDim.x
    stride = gw * bw

    j = i + stride
    aj = a[j]
    bj = b[j]

    # compute
    ci = core(ai, bi)
    cj = core(aj, bj)

    # write
    c[i] = ci
    c[j] = cj

@cuda.jit('void(float32[:], float32[:], float32[:])')
def vec_add_ilp_x4(a, b, c):
    # read
    i = cuda.grid(1)
    ai = a[i]
    bi = b[i]

    bw = cuda.blockDim.x
    gw = cuda.gridDim.x
    stride = gw * bw

    j = i + stride
    aj = a[j]
    bj = b[j]

    k = j + stride
    ak = a[k]
    bk = b[k]

    l = k + stride
    al = a[l]
    bl = b[l]

    # compute
    ci = core(ai, bi)
    cj = core(aj, bj)
    ck = core(ak, bk)
    cl = core(al, bl)

    # write 
    c[i] = ci
    c[j] = cj
    c[k] = ck
    c[l] = cl


@cuda.jit('void(float32[:], float32[:], float32[:])')
def vec_add_ilp_x8(a, b, c):
    # read
    i = cuda.grid(1)
    ai = a[i]
    bi = b[i]

    bw = cuda.blockDim.x
    gw = cuda.gridDim.x
    stride = gw * bw

    j = i + stride
    aj = a[j]
    bj = b[j]

    k = j + stride
    ak = a[k]
    bk = b[k]

    l = k + stride
    al = a[l]
    bl = b[l]

    m = l + stride
    am = a[m]
    bm = b[m]

    n = m + stride
    an = a[n]
    bn = b[n]

    p = n + stride
    ap = a[p]
    bp = b[p]

    q = n + stride
    aq = a[q]
    bq = b[q]

    # compute
    ci = core(ai, bi)
    cj = core(aj, bj)
    ck = core(ak, bk)
    cl = core(al, bl)

    cm = core(am, bm)
    cn = core(an, bn)
    cp = core(ap, bp)
    cq = core(aq, bq)

    # write
    c[i] = ci
    c[j] = cj
    c[k] = ck
    c[l] = cl

    c[m] = cm
    c[n] = cn
    c[p] = cp
    c[q] = cq


def time_this(kernel, gridsz, blocksz, args):
    timings = []
    cuda.synchronize()
    for i in range(10): # best of 10
        ts = timer()
        kernel[gridsz, blocksz](*args)
        cuda.synchronize()
        te = timer()
        timings.append(te - ts)
    return sum(timings) / len(timings)

def ceil_to_nearest(n, m):
    return int(math.ceil(n / m) * m)

def main():
    device = cuda.get_current_device()
    maxtpb = device.MAX_THREADS_PER_BLOCK
    warpsize = device.WARP_SIZE

    # benchmark loop
    vary_warpsize = []

    baseline = []
    ilpx2 = []
    ilpx4 = []
    ilpx8 = []

    approx_data_size = 10 * 2**20

    for multiplier in range(1, maxtpb // warpsize + 1):
        blksz = warpsize * multiplier
        gridsz = ceil_to_nearest(float(approx_data_size) / blksz, 8)
        print 'kernel config [%d, %d]' % (gridsz, blksz)

        N = blksz * gridsz
        A = np.arange(N, dtype=np.float32)
        B = np.arange(N, dtype=np.float32)

        print 'data size %dMB' % (N / 2.**20 * A.dtype.itemsize)

        dA = cuda.to_device(A)
        dB = cuda.to_device(B)

        assert float(N) / blksz == gridsz, (float(N) / blksz, gridsz)
        vary_warpsize.append(blksz)

        dC = cuda.device_array_like(A)
        basetime = time_this(vec_add, gridsz, blksz, (dA, dB, dC))
        expected_result = dC.copy_to_host()
        baseline.append(N / basetime)
        

        dC = cuda.device_array_like(A)
        x2time = time_this(vec_add_ilp_x2, gridsz//2, blksz, (dA, dB, dC))
        assert np.allclose(expected_result, dC.copy_to_host())
        ilpx2.append(N / x2time)

        dC = cuda.device_array_like(A)
        x4time = time_this(vec_add_ilp_x4, gridsz//4, blksz, (dA, dB, dC))
        assert np.allclose(expected_result, dC.copy_to_host())
        ilpx4.append(N / x4time)

        dC = cuda.device_array_like(A)
        x8time = time_this(vec_add_ilp_x8, gridsz//8, blksz, (dA, dB, dC))
        assert np.allclose(expected_result, dC.copy_to_host())
        ilpx8.append(N / x8time)

    pylab.plot(vary_warpsize, baseline, label='baseline')
    pylab.plot(vary_warpsize, ilpx2, label='ILP2')
    pylab.plot(vary_warpsize, ilpx4, label='ILP4')
    pylab.plot(vary_warpsize, ilpx8, label='ILP8')
    pylab.legend(loc=4)
    pylab.xlabel('block size')
    pylab.ylabel('float per second')
    pylab.show()


if __name__ == '__main__':
    main()
