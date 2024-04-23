from .mat import read_mat
import obspy
import numpy as np
from pathlib import Path


def read_data(path, format=None, convert_obspy=True, **kwargs):
    """
        Reads data with various formats supported by `obspy.read` and `.mat` format.

        Arguments:
            path : str : path to data
            format : str / None : data format to read. The data format can be inferred automatically, but sometimes it
                                  fails. Formats that are not always inferred properly (require explicit specification):
                                    - `SEG2`
            convert_obspy : bool : Whether to convert `obspy.Stream` to dict with Numpy.ndarray.
                                   Used for any format other than `.mat`

            kwargs : dict : other keyword arguments for `obspy.read`

        Returns:
            data : dict : Read data in dictionary. There are two keys: 'data' - numpy array of data,
                                                                       'hdrc' or 'stats' - header info

    """
    path = Path(path)
    if path.name[-4:] == '.mat':
        data = read_mat(path)
    else:
        data = read_by_obspy(path, format=format, **kwargs)
        if convert_obspy:
            data = convert_stream_to_dict(data)
    return data


def read_by_obspy(path, format=None, **kwargs):
    return obspy.read(path, format=format, **kwargs)


def list_to_object_numpy(list_obj, shape=None):
    N = len(list_obj)
    numpy_obj = np.empty(N, dtype=object)
    for i in range(N):
        numpy_obj[i] = list_obj[i]

    if shape is not None:
        numpy_obj = numpy_obj.reshape(shape)
    return numpy_obj


def convert_stream_to_dict(stream):
    return {'data': np.array([tr.data for tr in stream]),
            'stats': list_to_object_numpy([getattr(tr, 'stats', None) for tr in stream])}


def get_stats_by_index(stats, ind, shape=None):
    if stats is not None:
        return stats[ind]
    else:
        specifiers = ["{" f"0:0>{len(str(di))}d" "}" for di in shape]  # for keeping original sorting order in Stream
        return {"channel": "_".join(map(lambda x, y: y.format(x), ind, specifiers))}


def convert_dict_to_stream(data_in_dict):
    data = data_in_dict["data"]
    stats = data_in_dict.get("stats", None)
    if not isinstance(stats, np.ndarray) and (stats is not None):
        stats = list_to_object_numpy(stats)
    shape = data.shape[:-1]

    traces = [obspy.Trace(data=data[ind],
                          header=get_stats_by_index(stats, ind))
              for ind in np.ndindex(shape)]
    stream = obspy.Stream(traces)
    return stream
