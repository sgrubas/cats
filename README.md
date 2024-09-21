# Cluster Analysis of Trimmed Spectrograms (CATS)
CATS is a signal processing technique and framework for detecting and denoising sparse signals in the time-frequency domain. 
Particularly, very useful for processing earthquakes. 
This work is still in progress, and the package is under active development. 
Soon, here will be links to our papers/preprints.

## Key features of CATS
- **Versatile**. Any signals (not necessarily seismic) that are sparse in the time-frequency domain can be localized by CATS.
- **Flexible**. Any time-frequency transform can be used as a base (STFT, CWT, ...). Fast detection with STFT or more accurate denoising with CWT.
- **Fast** and **accurate**. Here will be links to our papers showing this.
- **Transparent** and **QC-friendly**. 
  - Minimum number of parameters which are easy to autotune.
  - Interpretable and visualizable workflow steps and parameters.
  - Collected cluster statistics can be used for custom post-processing and quality control (QC).


# Installation
To install the package:
1. Clone repository: `git clone https://github.com/sgrubas/cats.git`
2. Open the `cats` directory: `cd cats`
3. Install: 1) `pip install .` or 2) `pip install -e .` (editable mode)

## Dependencies
The package was tested on Python 3.9. See other dependencies in [requirements.txt](https://github.com/sgrubas/cats/blob/main/requirements.txt).

# Tutorials
- [Detection of seismic events](https://github.com/sgrubas/cats/blob/main/tutorials/DetectionTutorial.ipynb)
- [Autotuning CATS detector with Optuna](https://github.com/sgrubas/cats/blob/main/tutorials/DetectionAutotuner.ipynb)
- [Denoising seismic events](https://github.com/sgrubas/cats/blob/main/tutorials/DenoisingTutorial.ipynb)

# Demos:
## Signal detection with CATSDetector 
<img src="https://github.com/sgrubas/cats/blob/main/figures/DemoDetection_CATS.png" width="500"/>

## Signal denoising with CATSDenoiser and CATSDenoiserCWT
<img src="https://github.com/sgrubas/cats/blob/main/figures/DemoDenoising_CATS.png" width="400"/><img src="https://github.com/sgrubas/cats/blob/main/figures/DemoDenoising_CATS_CWT.png" width="400"/>

# Citation
If you find CATS useful for your research, please cite this repository (soon there will be links to our papers):
```
@article{grubas2023cats,
  title = {Cluster Analysis of Trimmed Spectrograms (CATS)},
  author = {Serafim Grubas and Mirko van der Baan},
  journal = {GitHub},
  url = {https://github.com/sgrubas/cats},
  year = {2023},
  doi = {TBC},
}
```

# Authors
- Serafim Grubas (serafimgrubas@gmail.com, grubas@ualberta.ca)
- Mirko van der Baan
