import numpy as np
import numba as nb
from functools import wraps

def ReshapeInputArray(dim : int, num : int = 1, methodfunc : bool = False, output : bool = True):
    mf = methodfunc * 1
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            d = dim - 1
            preshapes = []
            new_args = [args[0]] if mf else []
            for i, X in enumerate(args[mf : num + mf]):
                flatten = (X.ndim == d) 
                Y = np.expand_dims(X, axis=0) if flatten else X
                if i == 0: preshape = Y.shape[:-d] # First input shape remembering
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


def multiplex_axes(A, first_axis, second_axis):
    if second_axis is not None:
        ax1, ax2 = first_axis, second_axis
        ax1 = ax1 if ax1 >= 0 else A.ndim + ax1
        ax2 = ax2 if ax2 >= 0 else A.ndim + ax2
        d1, d2 = A.shape[ax1], A.shape[ax2]
        ax = ax1 + (ax2 > ax1)
        X = np.moveaxis(A, ax2, ax)
        new_ax = (ax - 1) * (ax > 1)
        X = X.reshape(X.shape[: new_ax] + (d1 * d2,) + X.shape[ax + 1:])
        return X, d2, new_ax
    else:
        return A, 1, first_axis

def check_dtype(x, dtype):
    if isinstance(dtype, tuple):
        dtypeness = bool(sum([x.dtype == dtpi for dtpi in dtype]))
    else:
        dtypeness = x.dtype == dtype
    return isinstance(x, np.ndarray) and dtypeness and x.flags.contiguous
