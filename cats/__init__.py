from .core import timefrequency
from .core import date
from .core import clustering
from .core import projection
from .core import association
from .core import utils
from .core import phaseseparation
from .core.plottingutils import plot_traces
from . import metrics
from .core.clustering import ClusterCatalogs
from .core.phaseseparation import SplinePhaseSeparator, TopoPhaseSeparator
from .detection import CATSDetector, CATSDetectionResult
from .denoising import CATSDenoiser, CATSDenoisingResult
from .denoising_cwt import CATSDenoiserCWT, CATSDenoisingCWTResult
from .data import import_sample_data, load_pretuned_CATS
from . import misc
from . import io
from . import tune
from .io import read_data
from .core import env_variables
from .core.env_variables import get_max_memory_available_for_cats, set_max_memory_for_cats


__version__ = "0.3.2"
