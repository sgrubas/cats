import numpy as np
import pkg_resources
import pickle
import zipfile
from typing import Literal
from cats import CATSDenoiser, CATSDetector
from cats import CATSDenoiser_CWT


def import_sample_data():
    zip_file = pkg_resources.resource_stream(__name__, "SampleDataset.zip")
    path = zip_file.name
    with zipfile.ZipFile(path) as zf:
        Data = {'data': np.load(zf.open('SampleDataset.npy'))}
        with zf.open('SampleDatasetMeta.pkl') as f:
            meta = pickle.load(f)
        Data.update(meta)
    return Data


def load_pretuned_CATS(mode: Literal["detector", "denoiser"] = "denoiser",
                       multitrace: bool = False,
                       time_frequency_base: Literal["STFT", "CWT"] = "STFT"):
    base = time_frequency_base
    cwt = (base == "CWT")
    name = 'm' * multitrace + "CATS"
    name = "_".join([name, base]) if cwt else name
    filename = "_".join([mode, name])

    assert mode in ["detector", "denoiser"], f"Unknown {mode = }, only {['detector', 'denoiser']} are available"
    assert base in ["STFT", "CWT"], f"Unknown {time_frequency_base = }, only {['STFT', 'CWT']} are available"
    if (mode == 'detector') and cwt:
        raise ValueError("CWT is available for `denoiser` mode only")

    file = pkg_resources.resource_stream(__name__, f"pretuned/{filename}.pickle").name

    if mode == 'denoiser':
        cats_model = CATSDenoiser.load(file) if not cwt else CATSDenoiser_CWT.load(file)
    else:
        if cwt:
            raise ValueError("CWT modification is available only for `denoiser` mode")
        cats_model = CATSDetector.load(file)

    print(f"Successfully loaded a pre-tuned CATS from {file}")

    return cats_model
