import numpy as np
from numba import cuda

#################################################
################ STA/LTA example ################
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
        

