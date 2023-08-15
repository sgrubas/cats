import numpy as np
import pkg_resources
import pickle
import zipfile


def import_sample_data():
    zip_file = pkg_resources.resource_stream(__name__, "SampleDataset.zip")
    path = zip_file.name
    with zipfile.ZipFile(path) as zf:
        Data = {'data': np.load(zf.open('SampleDataset.npy'))}
        with zf.open('SampleDatasetMeta.pkl') as f:
            meta = pickle.load(f)
        Data.update(meta)
    return Data
