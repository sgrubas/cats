import numpy as np
import pkg_resources, pickle


def import_sample_data():
    data_file = pkg_resources.resource_stream(__name__, "SampleDataset.npy")
    meta_file = pkg_resources.resource_stream(__name__, "SampleDatasetMeta.pkl")
    Data = {'data': np.load(data_file)}
    with open(meta_file.name, mode='rb') as f:
        meta = pickle.load(f)
    Data.update(meta)
    return Data