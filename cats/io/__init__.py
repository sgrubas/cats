import obspy

from .mat import write_mat, read_mat, HDRC_FIELDS_INFO
from obspy import read as obspy_read
import numpy as np
from pathlib import Path


def read_data(path, format=None, **kwargs):
    """
        Reads data with various formats supported by `obspy.read` and `.mat` format.

        Arguments:
            path : str : path to data
            format : str / None : data format to read. The data format can be inferred automatically, but sometimes it
                                  fails. Formats that are not always inferred properly (require explicit specification):
                                    - `SEG2`

        Returns:
            data : dict : Read data in dictionary. There are two keys: 'data' - numpy array of data,
                                                                       'hdrc' or 'stats' - header info

    """
    path = Path(path)
    if path.name[-4:] == '.mat':
        return read_mat(path)
    else:
        return _read_by_obspy(path, format=format, **kwargs)


def _read_by_obspy(path, format=None, **kwargs):
    stream = obspy_read(path, format=format, **kwargs)
    return convert_stream_to_dict(stream)


def convert_stream_to_dict(stream):
    return {'data': np.array([tr.data for tr in stream]),
            'stats': [getattr(tr, 'stats', None) for tr in stream]}


def convert_numpy_to_stream(data, headers=None):
    ...
    # return obspy.Stream
