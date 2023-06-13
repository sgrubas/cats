from .mat import write_mat, read_mat, HDRC_FIELDS_INFO
from obspy import read
import numpy as np
from pathlib import Path


def read_data(path):
    path = Path(path)
    if path.name[-4:] == '.mat':
        return read_mat(path)
    else:
        return _read_by_obspy(path)


def _read_by_obspy(path):
    stream = read(path)
    data_dict = {'data': np.array([tr.data for tr in stream]),
                 'stats': getattr(stream, 'stats', None)}
    return data_dict
