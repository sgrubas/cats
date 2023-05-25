"""
    Implements read and write functions for `.mat` files:
        - read_mat
        - write_mat
    For header documentation see `cats.io.mat.HDRC_FIELDS_INFO`
"""

from scipy.io import savemat, loadmat


HDRC_FIELDS_INFO = {
    # Experiment data
    'b':         'begin time (sec.)',
    'e':         'end time (sec.)',
    'kzdate':    'reference date (MMM DD, YYYY) (ex: Nov 20, 1999)',
    'kztime':    'reference time (HH:MM:SS.fff) (ex: 00:12:55.840)',
    'timez':     'time zone',
    'iztype':    'type of reference time (UTC, GMT, ...)',
    'kinst':     'generic name of recording instruments',
    'knetwk':    'name of seismic network (or experiment name)',
    'dattype':   'type of data (microseismic, VSP, continuous, triggered, ...)',
    'wellinj':   'name of injection well',
    'wellrec':   'name of recording well',

    # Data metadata
    'kstnm':     'receiver names',
    'stnb':      'number of receivers',
    'staz':      'azimuth of first horizontal component, clockwise from North (deg.)',
    'stinc':     'incidence angle of horizontal components from vertical (deg.)',
    'cmpo':      'components order (ex: h1 h2 z)',
    'stla':      'receiver latitude (decimal deg., North positive)',
    'stlo':      'receiver longitude (decimal deg., East positive)',
    'stutmx':    'receiver utm x (m)',
    'stutmy':    'receiver utm y (m)',
    'stel':      'receiver elevation above sea level (m)',
    'stdp':      'receiver depth below surface (m)',
    'npts':      'number of data points',
    'delta':     'sampling interval (sec.)',
    'odelta':    'observed sampling interval (sec.) if different from `delta`',
    'idep':      'units of the data (m/s, m, counts, volts, ...)',
    'scale':     'descaling factor (from time series units to physical units m/s)',
    'pozero':    'poles-zeros information for instruments frequency response',
    'datlev':    'level in processing of the current data (raw, filtered, oriented, ...)',
    'comm':      'comments (including point of origin, z positive up or down, type of coordinate system)',

    # Event metadata
    'ppick':        'time of direct P-wave arrival (in sec. relative to reference time)',
    'spick':        'time of S-wave arrival (in sec. relative to reference time)',
    'ppick_a':      'amplitude of P-wave arrival (in raw data units unless specified `comm`)',
    'spick_a':      'amplitude of S-wave arrival (in raw data units unless specified `comm`)',
    'ppick_qual':   'quality of P-wave picks (4: very good, 3: good, 2: satisfactory, 1: problematic)',
    'spick_qual':   'quality of S-wave picks (4: very good, 3: good, 2: satisfactory, 1: problematic)',
    'kevnm':        'event name',
    'evla':         'event latitude (decimal deg., North positive)',
    'evlo':         'event longitude (decimal deg., East positive)',
    'evdp':         'event depth below surface (m)',
    'evutmx':       'event utm x (m)',
    'evutmy':       'event utm y (m)',
    'evaz':         'azimuths calculated around p-wave picks for each receiver (deg.) (=backazimuths)',
    'mag':          'event magnitude',
    'imagtyp':      'magnitude type (IMB, IMS, IML, IMW, IMD, ...)',
    'evdate':       'event origin date (MMM DD, YYYY) (ex: Nov 20, 1999)',
    'evtime':       'event origin time (HH:MM:SS.fff) (ex: 00:12:55.840)',
    'ievtype':      'type of event (ims=microseismic event, plus events defined in SAC like: ieq, iqb, ...)',
    'foc':          'focal mechanism tensor stored as a 3x3 array',
    'rms':          'rms residuals from absolute location',
    'unc':          'uncertainties in absolute location ([uncx uncy, uncz])',
    'parentsname':  'name and path of file from which the event file is coming from',
    'parentstime':  'beginning time of event data file in `parentsname` data file',
}


def write_mat(path, mat_dict, compress=True):
    """
        Saves the mat file as `.mat`.

        Arguments:
            path : str / file-like object : filepath name
            mat_dict: dictionary containing `data` and `hdrc`.
                    For `hdrc` documentation see `cats.io.mat.HDRC_FIELDS_INFO`
            compress: boolean : whether to compress matrix data

    """

    savemat(path, mat_dict, do_compression=compress)


def read_mat(path):
    """
        Reads the mat file into python dictionary.
        Main fields: `hdrc` - header, `data` - data

        Arguments:
            path : str / file-like object : filepath name with `.mat` file
    """
    return loadmat(path, simplify_cells=True)
