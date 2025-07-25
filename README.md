# Cluster Analysis of Trimmed Spectrograms (CATS)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15627707.svg)](https://zenodo.org/doi/10.5281/zenodo.15627707)

CATS is a signal processing technique and framework for detecting and denoising sparse signals in the time-frequency domain. 
Particularly, very useful for processing earthquakes. 
This work is still in progress, and the package is under active development. 
Soon, here will be links to our papers/preprints.

## Key features of CATS
- **Versatile**. Any sparse signals in the time-frequency domain can be localized by CATS.
- **Flexible**. Fast detection with STFT or more accurate denoising with CWT.
- **Fast** and **accurate**. Here will be links to our papers showing this.
- **Comprehensive quality control**. 
  - Autotunable parameters with direct physical interpretation.
  - Easy visualization of all intermediate workflow steps.
  - Collected cluster statistics allow for fine-grained QC and classification of signals.


# Installation
To install the package:
1. Short way: `pip install git+https://github.com/sgrubas/cats.git`
2. Other way:
   1. Clone repository: `git clone https://github.com/sgrubas/cats.git`
   2. Open the `cats` directory: `cd cats`
   3. Install: 1) `pip install .` or 2) `pip install -e .` (editable mode)
3. To update: `pip install -U git+https://github.com/sgrubas/cats.git`

## Dependencies
The package was tested on Python 3.9. See other dependencies in [requirements.txt](https://github.com/sgrubas/cats/blob/main/requirements.txt).

# Tutorials
- [Detection of seismic events](https://github.com/sgrubas/cats/blob/main/tutorials/DetectionTutorial.ipynb)
- [Autotuning CATS detector with Optuna](https://github.com/sgrubas/cats/blob/main/tutorials/DetectionAutotuner.ipynb)
- [Denoising seismic events](https://github.com/sgrubas/cats/blob/main/tutorials/DenoisingTutorial.ipynb)
- [Autotuning CATS denoising with Optuna](https://github.com/sgrubas/cats/blob/main/tutorials/DenoisingAutotuner.ipynb)

# Demos:
## Signal detection with CATSDetector 
<img src="https://github.com/sgrubas/cats/blob/main/figures/DemoDetection_CATS.png" width="500"/>

## Signal denoising with CATSDenoiser and CATSDenoiserCWT
<img src="https://github.com/sgrubas/cats/blob/main/figures/DemoDenoising_CATS.png" width="400"/><img src="https://github.com/sgrubas/cats/blob/main/figures/DemoDenoising_CATS_CWT.png" width="400"/>

# Citation
If you find CATS useful for your research, please cite the repository ([CITATION.bib](https://github.com/sgrubas/cats/blob/main/CITATION.bib)).

# Authors
- Serafim Grubas (serafimgrubas@gmail.com, grubas@ualberta.ca)
- Mirko van der Baan
