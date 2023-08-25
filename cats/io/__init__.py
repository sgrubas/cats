from .mat import write_mat, read_mat, HDRC_FIELDS_INFO
from obspy import read as obspy_read
import numpy as np
from pathlib import Path


def read_data(path, format=None):
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
        return _read_by_obspy(path, format=format)


def _read_by_obspy(path, format=None):
    stream = obspy_read(path, format=format)
    data_dict = {'data': np.array([tr.data for tr in stream]),
                 'stats': getattr(stream[0], 'stats', None)}
    return data_dict
