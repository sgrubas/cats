from setuptools import setup, find_packages

NAME            = "CATS"
VERSION         = "0.1.0"
DESCRIPTION     = "Cluster Analysis of Trimmed Spectrograms (CATS)"
URL             = "https://github.com/sgrubas/CATS"
LICENSE         = "MIT"
AUTHOR          = "Serafim Grubas, Mirko van der Baan"
EMAIL           = "serafimgrubas@gmail.com, grubas@ualberta.ca"
KEYWORDS        = ["Clustering", "Detection", "Denoising", "Noise estimation", 
                   "Spectrogram", "Time-Frequency", "Sparse", "Earthquake", "Voice"]
CLASSIFIERS     = [
                    "Development Status :: Beta",
                    "Intended Audience :: Geophysicist",
                    "Natural Language :: English",
                    f"License :: {LICENSE}",
                    "Operating System :: OS Independent",
                    "Programming Language :: Python :: 3.9",
                    "Topic :: Scientific/Engineering",
                    ]
INSTALL_REQUIRES = [
                    'numpy',
                    'numba',
                    'scipy',
                    'ssqueezepy',
                    'matplotlib',
                    'holoviews',
                    'xarray',
                    'colorednoise'
                    ]
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    maintainer=AUTHOR,
    maintainer_email=EMAIL,
    classifiers=CLASSIFIERS,
    keywords=KEYWORDS,
    packages=find_packages(),
    # package_dir={"": NAME},
    url=URL,
    zip_safe=False,
    install_requires=INSTALL_REQUIRES,
    # include_package_data=True,
    package_data={NAME: ["data/*.npy", "data/*.pkl"]}
    )

