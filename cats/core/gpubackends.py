import numpy as np
from numba import cuda

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

