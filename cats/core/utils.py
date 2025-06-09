import numpy as np
import numba as nb
from functools import wraps
from pydantic import BaseModel, Extra
from typing import Union, Dict, List, Set, Tuple
from timeit import default_timer


#  Generic decorator for formatting inputs & outputs of function when N-dimensional array
#  must be reshaped to predefined number of dimensions `dim`
def _ReshapeInputs(inputs, dim, input_num, methodfunc):
    mf = methodfunc * 1
    d = dim - 1
    input_num = input_num if input_num >= 0 else len(inputs)
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
            preshape_i = preshapes[i * (not first_shape)]
            zi = zi.reshape(*(preshape_i + postshape))  # reshaping back to the input shape
            zndim = inputs[mf + i * (not first_shape)].ndim
            zi = zi.squeeze() if zndim == d else zi
            V.append(zi)
        V = tuple(V) + tuple(outputs[out_num:])
    return V


def ReshapeArraysDecorator(dim: int, input_num: int = 1, methodfunc: bool = False,
                           output_num: int = 1, first_shape: bool = True):
    """
        Decorator for reshaping input **positional arguments only** and output arrays to a predefined number
        of dimensions `dim`. The last axis is always main axis over which all the computations are performed.

        Arguments:
            dim : int : number of dimensions to be used for computations
            input_num : int : number of first positional arguments (arrays) to be reshaped
            methodfunc : boolean : whether it is applied to a method of class or a standalone function
            output_num : int : number of first outputs (arrays) to be reshaped back to original shape
            first_shape : boolean : whether to use the shape of the first input array for reshaping outputs back
    """
    def decorator(function):
        @wraps(function)
        def wrapper(*inputs, **kwargs):
            preshapes, new_args = _ReshapeInputs(inputs, dim, input_num, methodfunc)
            outputs = function(*tuple(new_args), **kwargs)  # Executing the function
            Z = _ReshapeOutputs(outputs, dim, output_num, methodfunc, first_shape, preshapes, inputs)
            return Z
        return wrapper
    return decorator


class StatusMessenger(BaseModel, extra=Extra.allow):
    verbose: bool
    operation: str

    def __enter__(self):
        self.status = '...'
        self.t0 = default_timer()
        if self.verbose:
            print(f"{self.operation}\t{self.status}", end='\t')

    def __exit__(self, *args):
        self.dt = float('%.3g' % (default_timer() - self.t0))
        self.status = f'Completed in {self.dt} sec'

        if self.verbose:
            print(self.status)


class StatusKeeper(BaseModel, extra=Extra.allow):
    verbose: bool
    wait_message: str = '...'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_process = None
        self.history = {'Total': 0.0}

    def __call__(self, current_process):
        self.current_process = current_process
        return self

    def __enter__(self):
        self.history[self.current_process] = default_timer()
        if self.verbose:
            print(f"{len(self.history) - 1}. {self.current_process}\t{self.wait_message}", end='\t')

    def __exit__(self, *args):
        self.history[self.current_process] = dt = default_timer() - self.history[self.current_process]
        self.history['Total'] += dt
        if self.verbose:
            print(f"Completed in {float('%.3g' % dt)} sec")

    def print_total_time(self):
        if self.verbose:
            print(f"Total elapsed time:\t{float('%.3g' % self.history['Total'])} sec\n")

    def merge(self, other):
        for proc, dt in other.history.items():
            self.history[proc] += dt


@nb.njit("i8[:, :](i8, i8)", cache=True)
def get_interval_division(N, L):
    """
    `J` will be adjusted so that `N % J --> min` where the last frame will have `J + N % J`
    """
    assert L > 0

    if N < L: 
        return np.array([[0, N - 1]])
    else:
        n, g = divmod(N, L)
        l, g = divmod(g, n)
        M = L + l
        inds = np.array([0] + [M] * (n - 1) + [M + g]).cumsum()
        intervals = np.stack((inds[:-1], inds[1:] - 1), axis=1)
        return intervals


@nb.njit("i8[:, :](i8, f8, i8)", cache=True)
def get_logarithmic_interval_division(N, log_step, log_base):
    intervals = [[0, 0]]
    i2 = 0
    while i2 < N - 1:
        i1 = i2 + 1
        i2 = min(round(i1 * log_base ** log_step), N - 1)
        intervals.append([i1, i2])
    return np.array(intervals)


def _scalarORarray_to_tuple(d, minsize):
    if np.isscalar(d):
        d = (int(d),) * minsize
    else: 
        assert isinstance(d, (tuple, list, np.ndarray)) and len(d) >= minsize
        d = tuple(int(di) for di in d)
    return d


def count_slice_objects(tuple_sequence):
    return sum([1 for ii in tuple_sequence if isinstance(ii, slice)])


def replace_int_by_list(ind):
    if isinstance(ind, (list, tuple)):
        return tuple(map(lambda i: [i] if isinstance(i, int) else i, ind))
    elif isinstance(ind, int):
        return [ind]
    elif isinstance(ind, slice):
        return ind
    else:
        raise ValueError(f"Invalid type of `ind` {type(ind) = }")


def replace_int_by_slice(ind, ind2slice):
    if ind2slice is not None:
        ind2slice = ind2slice if isinstance(ind2slice, (tuple, list)) else [ind2slice]

        if isinstance(ind, (list, tuple)):
            assert max(ind2slice) < len(ind), "Index for slice is out of range"
            return tuple(map(lambda x: slice(None) if (x[0] in ind2slice) else x[1], enumerate(ind)))
        elif isinstance(ind, int):
            assert max(ind2slice) < 1, "Index for slice is out of range"
            return slice(None) if (0 in ind2slice) else ind
        else:
            raise ValueError(f"Invalid type of `ind` {type(ind) = }")
    else:
        return ind


def give_trace_dim_names(ndims, aggr_clustering_axis=None, trace_dim_names=None):

    if trace_dim_names is None:
        if aggr_clustering_axis is not None:
            axes = aggr_clustering_axis if isinstance(aggr_clustering_axis, (tuple, list)) else [aggr_clustering_axis]
            assert (len(axes) <= ndims) and max(axes) < ndims, "Number of `aggr_clustering_axis` must match `ndims`"
            names = [("Aggregated" if i in axes else "Trace") + f"_dim_{i}"
                     for i in range(ndims)]
            if len(axes) == 1:
                names[axes[0]] = "Component"
                if ndims - len(axes) == 1:
                    names[1 - axes[0]] = "Station"
            return names
        else:
            return [f"Trace_dim_{i}" for i in range(ndims)]
    else:
        assert isinstance(trace_dim_names, (list, tuple)) and len(trace_dim_names) == ndims, \
            "Trace dim names must be list or tuple, and length must be equal to number of dimensions"

        return trace_dim_names


def format_index_by_dimensions(ind, shape, slice_dims, default_ind=0):
    ndim = len(shape) - slice_dims
    if ind is None:
        ind = format_index_by_dimensions((default_ind,) * ndim, shape, slice_dims)
    elif isinstance(ind, (int, slice)):
        ind = format_index_by_dimensions((ind,) + (default_ind,) * (ndim - isinstance(ind, int)),
                                         shape, slice_dims)
    elif isinstance(ind, (list, tuple)):
        ind = tuple(ind)
        slice_count = count_slice_objects(ind)
        ind = ind + (default_ind,) * (ndim - len(ind) + slice_count)
        ind = ind + (slice(None),) * (slice_dims - slice_count)
        assert len(ind) == len(shape)
    else:
        raise KeyError("Unknown 'ind' type")

    return ind


def format_index_by_dimensions_new(ind, ndim, default_ind=0):
    if ind is None:
        ind = (default_ind,)

    if isinstance(ind, (int, slice)):
        ind = format_index_by_dimensions_new((ind,), ndim, default_ind)
    elif isinstance(ind, (list, tuple)):
        ind = tuple(ind)
        slice_dims = ndim - len(ind)
        ind = ind + (slice(None),) * slice_dims
        assert len(ind) == ndim
    elif isinstance(ind, type(...)):
        ind = (ind,)
    else:
        raise KeyError(f"Unknown ind type {type(ind) = }")

    return ind


def format_interval_by_limits(interval, limits):
    if isinstance(interval, tuple):
        t1 = limits[0] if (y := interval[0]) is None else y
        t2 = limits[1] if (y := interval[1]) is None else y
    elif interval is None:
        t1, t2 = limits
    else:
        t1, t2 = interval

    interval = tuple(map(lambda x: np.clip(x, *limits) if x is not None else x, (t1, t2)))
    # interval = tuple(np.clip((t1, t2), *limits))
    return interval


def give_index_slice_by_limits(interval, dt, t0=0):
    t1, t2 = interval
    start = t1 if t1 is None else round((t1 - t0) / dt)
    end = t2 if t2 is None else round((t2 - t0) / dt) + 1
    return slice(start, end)


def cast_to_bool_dict(iterable: Union[bool, List[str], Tuple[str], Set[str], Dict[str, bool]],
                      reference_keys: Union[list, tuple, set]):
    if isinstance(iterable, bool):
        output_dict = dict.fromkeys(reference_keys, iterable)
    elif isinstance(iterable, (list, tuple, set, dict)):
        output_dict = iterable if isinstance(iterable, dict) else dict.fromkeys(iterable, True)
        for kw in reference_keys:
            output_dict.setdefault(kw, False)
    else:
        raise TypeError(f"Unknown input data type of `iterable` - {type(iterable)}, must be [bool, dict, list]")
    return output_dict


def del_vals_by_keys(dict_vals: dict,
                     dict_cond: Dict[str, bool],
                     keys: Union[list, tuple]):
    available_keys = dict_vals.keys()
    for kw in keys:
        if (not dict_cond[kw]) and (kw in available_keys):
            del dict_vals[kw]


AGGREGATORS = {"min": np.min,
               "mean": np.mean,
               "max": np.max,
               "median": np.median,
               "sum": np.sum,
               "prod": np.prod,
               "any": np.any,
               "all": np.all}


def aggregate_array_by_axis_and_func(array, axis, func, min_last_dims):
    if (array.ndim > min_last_dims) and (axis is not None) and (func is not None):
        max_axis = axis if isinstance(axis, int) else max(axis)  # max - to check if 'min_last_dims' axes are involved
        assert max_axis < array.ndim - min_last_dims  # last axes must not be aggregated
        func = func if callable(func) else AGGREGATORS[func]
        array = func(array, axis=axis, keepdims=True)
    return array


@nb.njit(["List(UniTuple(f8, 4))(f8[:, :], f8, f8)",
          "List(Tuple((i8, f8, i8, f8)))(i8[:, :], f8, f8)"])
def _give_rectangles(events, yloc, dy):
    rectangles = []
    for t1, t2 in events:
        rectangles.append((t1, yloc - dy, t2, yloc + dy))
    return rectangles


def give_rectangles(events, yloc, dy):
    rectangles = []
    if len(events) > 0:
        for trace, yi in zip(events, yloc):
            if len(trace) > 0:
                rectangles += _give_rectangles(trace, yi, dy)
    return rectangles


def update_object_params(obj, **params):
    """
        Updates the object instance with changed parameters by calling `_set_params()` method
    """
    for attribute, value in params.items():
        if hasattr(obj, attribute):
            setattr(obj, attribute, value)
        else:
            raise AttributeError(f'{type(obj)} has no attribute: {attribute}')
    obj._set_params()


def give_nonzero_limits(array, initials=(1e-1, 1e1)):
    arr_pos = array[array > 0]
    zerosize = arr_pos.size == 0
    cmin = initials[0] if zerosize else arr_pos.min()
    cmax = initials[1] if zerosize else arr_pos.max()
    return cmin, cmax


@nb.vectorize(["f8(c16)", "f4(c8)"], cache=True)
def complex_abs_square(x):
    return x.real ** 2 + x.imag ** 2


def intervals_intersection_inds(intervals_array, reference_interval):
    t1, t2 = reference_interval
    interval_inside = (t1 <= intervals_array) & (intervals_array <= t2)
    interval_inside = interval_inside[:, 0] | interval_inside[:, 1]
    interval_outside = (intervals_array[:, 0] <= t1) & (t2 <= intervals_array[:, 1])  # wider than (t1, t2)
    interval_inds = interval_inside | interval_outside
    return interval_inds


def intervals_intersection(intervals_array, reference_interval):
    if len(intervals_array) > 0:
        inds = intervals_intersection_inds(intervals_array, reference_interval)
        return intervals_array[inds]
    else:
        return np.zeros((0, 2))


def to2d_array_with_num_columns(array, num_columns=2):
    if array.size == 0:
        arr = array.reshape((0, num_columns))
    else:
        arr = np.expand_dims(array, 0) if array.ndim == 1 else array
    return arr


def mat_structure_to_tight_dataframe_dict(mat_struct):
    adpt_mat = {}
    len_cols = len(mat_struct.columns)
    for name in mat_struct._fieldnames:
        attr = np.array(getattr(mat_struct, name), ndmin=1)
        if name == 'data':
            attr = to2d_array_with_num_columns(attr, num_columns=len_cols)
        adpt_mat[name] = attr.tolist()
    adpt_mat['columns'] = [col.replace(" ", "") for col in adpt_mat['columns']]
    return adpt_mat


def make_default_index_if_outrange(tuple_ind, shape, default_ind_value=0):
    fixed_ind = tuple(map(lambda x, y: default_ind_value if x >= y else x, tuple_ind, shape))
    return fixed_ind


def make_default_index_on_axis(tuple_ind, ax, default_ind_value=0):
    if ax is not None:
        ind = list(tuple_ind)
        ind[ax] = default_ind_value
        ind = tuple(ind)
    else:
        ind = tuple_ind
    return ind
