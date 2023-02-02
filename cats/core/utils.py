import numpy as np
import numba as nb
from functools import wraps

#  Generic decorator for formatting inputs & outputs of function when N-dimensional array
#  must be reshaped to predefined number of dimensions `dim`
def ReshapeInputArray(dim : int, num : int = 1, methodfunc : bool = False, output : bool = True):
    mf = methodfunc * 1
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            d = dim - 1
            new_args = [args[0]] if mf else []
            for i, X in enumerate(args[mf : num + mf]):
                flatten = (X.ndim == d) 
                Y = np.expand_dims(X, axis=0) if flatten else X
                if i == 0:
                    preshape = Y.shape[:-d] # First input shape remembering
                new_args.append(Y.reshape(-1, *Y.shape[-d:])) # Reshaping to `dim`-dimensional
            new_args += list(args[num + mf:])
            ############
            Z = function(*tuple(new_args), **kwargs) # Executing the function (one output only)
            ############
            if output:
                postshape = Z.shape[1:]  
                Z = Z.reshape(*(preshape + postshape)) # reshaping back to the input shape
                Z = Z.squeeze() if args[mf].ndim == d else Z
            return Z
        return wrapper
    return decorator

@nb.njit("i8[:, :](i8, i8)")
def get_interval_division(N, L):
    """
    `J` will be adjusted so that `N % J --> min` where the last frame will have `J + N % J`
    """
    if N < L: 
        return np.array([[0, N]])
    else:
        n, g = divmod(N, L)
        l, g = divmod(g, n)
        M = L + l
        inds = np.array([0] + [M] * (n - 1) + [M + g]).cumsum()
        intervals = np.stack((inds[:-1], inds[1:]), axis=1)
        return intervals


def _scalarORarray_to_tuple(d, minsize):
    if np.isscalar(d):
        d = (int(d),) * minsize
    else: 
        assert isinstance(d, (tuple, list, np.ndarray)) and len(d) >= minsize
        d = tuple(int(di) for di in d)
    return d

