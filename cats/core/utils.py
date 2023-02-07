import numpy as np
import numba as nb
from functools import wraps

#  Generic decorator for formatting inputs & outputs of function when N-dimensional array
#  must be reshaped to predefined number of dimensions `dim`
def _ReshapeInputs(inputs, dim, input_num, methodfunc):
    mf = methodfunc * 1
    d = dim - 1
    new_args = [inputs[0]] if mf else []
    preshapes = []
    for i, X in enumerate(inputs[mf: input_num + mf]):
        flatten = (X.ndim == d)
        Y = np.expand_dims(X, axis=0) if flatten else X
        preshapes.append(Y.shape[:-d])  # input shape remembering
        new_args.append(Y.reshape(-1, *Y.shape[-d:]))  # Reshaping to `dim`-dimensional
    new_args += list(inputs[input_num + mf:])
    return preshapes, new_args


def _ReshapeOutputs(outputs, dim, output_num, methodfunc, first_shape, preshapes, inputs):
    mf = methodfunc * 1
    d = dim - 1
    if output_num == 0:
        V = outputs
    elif (output_num > 0) and (not isinstance(outputs, tuple)):
        postshape = outputs.shape[1:]
        Z = outputs.reshape(*(preshapes[0] + postshape))  # reshaping back to the input shape
        V = outputs.squeeze() if inputs[mf].ndim == d else Z
    else:
        out_num = min(len(outputs), output_num)
        V = []
        for i, zi in enumerate(outputs[: out_num]):
            postshape = zi.shape[1:]
            preshape_i = preshapes[i * first_shape]
            zi = zi.reshape(*(preshape_i + postshape))  # reshaping back to the input shape
            zndim = inputs[mf + i * first_shape].ndim
            zi = zi.squeeze() if zndim == d else zi
            V.append(zi)
        V = tuple(V)
    return V


def ReshapeArraysDecorator(dim : int, input_num : int = 1, methodfunc : bool = False,
                  output_num : int = 1, first_shape : bool = True):
    def decorator(function):
        @wraps(function)
        def wrapper(*inputs, **kwargs):
            preshapes, new_args = _ReshapeInputs(inputs, dim, input_num, methodfunc)
            outputs = function(*tuple(new_args), **kwargs)  # Executing the function
            Z = _ReshapeOutputs(outputs, dim, output_num, methodfunc, first_shape, preshapes, inputs)
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

