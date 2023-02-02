import numpy as np
from numba import cuda
import cmath


#######################################################
#################### PSD algorithm ####################
#######################################################

@cuda.jit('f4[:,:,:], f4[:,:], f4[:], i8, f8, i8')
def kernelPSD(Y, X, w, L, c, step):
    i, j, k = cuda.grid(3)
    
    if i < Y.shape[0] and j < Y.shape[1] and k < Y.shape[2]:        
        yi = 0.0j
        for m in range(L):
            yi += w[m] * X[i, j * step + m] * cmath.exp(-2j * cmath.pi * m * k / L)
        Y[i, j, k] = c * abs(yi)**2
        if k == 0 or k == Y.shape[2] - 1:
            Y[i, j, k] /= 2

def gpuPSD_API(x, w, c, Nt, Nf, step, threadsperblock=None, **kwargs):
         
    # Constants
    L = len(w)
    Nx = len(x)
    
    # Sending arrays to GPU
    stream = cuda.stream()
    Xd = cuda.to_device(np.float32(x), stream=stream)
    wd = cuda.to_device(np.float32(w), stream=stream)
    
    # final array is created at gpu
    Yd = cuda.device_array((Nx, Nt, Nf), dtype=np.float32)
        
    # GPU settings
    if threadsperblock is None:
        threadsperblock = (32, 32, 1)
    blockspergrid = tuple(int(np.ceil(Yd.shape[i] / threadsperblock[i]))
                          for i in range(len(Yd.shape)))
    
    # PSD computation
    kernelPSD[blockspergrid, threadsperblock](Yd, Xd, wd, L, c, step)

    # Sending final array to CPU
    Y = Yd.copy_to_host()
    return Y


#################################################
#################### STA/LTA ####################
#################################################


@cuda.jit('float32[:,:], float32[:,:], int64, int64, int64, int64, int64')
def gpu_STA_LTA_kernel(Y, X, short, long, center, sta_left, sta_right):

    i, j = cuda.grid(2)
    
    if i < Y.shape[0] and j < Y.shape[1] - sta_right:
        xi = X[i]
        sta, lta = 0.0, 0.0 
        for k in range(long):
            lta += xi[j + k]
            if k < short:
                sta += xi[j + sta_left + k]
        Y[i, j + center] += sta / (lta + 1e-16)
        
def gpu_STA_LTA_API(X, long, center, sta_left, sta_right, threadsperblock=None):
    
    # Sending arrays to GPU
    stream = cuda.stream()
    Xd = cuda.to_device(np.float32(X), stream=stream)
    
    # final array is created at gpu
    Yd = cuda.device_array(X.shape, dtype=np.float32)
        
    # GPU settings
    if threadsperblock is None:
        threadsperblock = (32, 32)
    blockspergrid = tuple(int(np.ceil(Yd.shape[i] / threadsperblock[i]))
                          for i in range(len(Yd.shape)))
    
    short = sta_right - sta_left
    # PSD computation
    gpu_STA_LTA_kernel[blockspergrid, threadsperblock](Yd, Xd, short, long, center, sta_left, sta_right)

    # Sending final array to CPU
    Y = Yd.copy_to_host()
    return Y

