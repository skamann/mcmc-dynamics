import contextlib
import logging
import uuid
import numpy as np
import pandas as pd
import cjam
try:
    from importlib.resources import files
except ImportError:  # for Python v<3.9
    from importlib_resources import files
from multiprocessing import Pool
from scipy import stats
from astropy import units as u
from astropy.table import Table

from ..runner import Runner
from ... import config
from ...parameter import Parameters
from ...utils.coordinates import calc_xy_offset
from ...utils.files import MgeReader, get_mge, get_nearest_neigbhbour_idx2
from ...utils.morphology.deprojection import find_barq_limits

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


def init_cjam(x, y, mge_mass, mge_lum, *args):
    """
    Initializer for running a CJAM model.

    This function declares global variables containing the observables used by
    the CJAM code.

    Parameters
    ----------
    x : astropy Quantity
        The x-coordinates of the velocity data, relative to the cluster centre.
    y : astropy Quantity
        The x-coordinates of the velocity data, relative to the cluster centre.
    mge_mass : astropy QTAble
        The MGE profile used to describe the mass density.
    mge_lum : astropy QTable
        The MGE profile used to describe the tracer density.
    args
        This function does not use any additional arguments.
    """
    global gx
    global gy
    global gmge_mass
    global gmge_lum

    gx = x
    gy = y
    gmge_mass = mge_mass
    gmge_lum = mge_lum


def run_cjam(parameters):
    """
    Helper function for running a single CJAM model for a given set of
    parameters.

    Parameters
    ----------
    parameters : dict
       The model input. The dictionary needs to contain at least the following
       keys:
       * d - The distance to the cluster
       * beta - The anisotropy parameter
       * kappa - The rotation parameter, either as a global value or per
                 tracer MGE component.
       * mlr - The mass-to-light ratio, either as a global value of per mass
               MGE component.
       * incl - The inclination of the model.
       * mbh - The assumed mass of the central black hole.
       * rbh - The assumed radius of the black hole component included in the
               model.
       As an optional key, `mge_filename` is used, which should point to an
       ESCV-file from which the MGE profiles are loaded.

    Returns
    -------
    vz : array_like
        The first order moments along the line of sight calculated by the CJAM
        code.
    v2zz : array_like
        The second order moments along the line of sight calculated by the
        CJAM code.
    """
    global gx, gy, gmge_mass, gmge_lum
    
    use_mge_grid = gmge_mass is None
    if use_mge_grid:
        mge_filename = parameters['mge_filename']
        mge_lum, mge_mass = get_mge(mge_filename)
        mge_lum = mge_lum.data
        mge_mass = mge_mass.data
    
    else:
        mge_lum = gmge_lum
        mge_mass = gmge_mass

    # shifting the centre and rotating coords is not allowed here,
    # since this function is used only with fake data

    model = cjam.axisymmetric(gx, gy, mge_lum, mge_mass, parameters['d'], beta=parameters['beta'],
                              kappa=parameters["kappa"], mscale=parameters['mlr'].value, incl=parameters['incl'],
                              mbh=parameters['mbh'], rbh=parameters['rbh'])

    # get velocity and dispersion at every data point
    # note that astropy Quantities cannot be pickled, so multiprocessing only works when the values are returned
    return model['vz'].value, model['v2zz'].value


class Axisymmetric(Runner):
    """
    This class implements the most basic axisymmetric Jeans models, with a
    constant mass-to-light ratio and a constant rotation parameter kappa.

    The class uses the following model parameters, which can be optimized
    during the analysis.

    (1) d - The distance to the system.
    (2) mlr - The global mass-to-light ratio. Note that this parameter is
    degenerate with the distance `d` unless proper motions are available.
    (3) barq - The intrinsic flattening of the system, as defined in
    Watkins et al. (2013)
    (4) kappa_x - The x-component of the rotation parameter.
    (5) kappa_y - The y-component of the rotation parameter.
    (6) beta - The anisotropy parameter.
    (7) mbh - The mass of the central black hole.
    (8) ra_center - The right ascension coordinate of the cluster centre.
    (9) dec_center - The declination coordinate of the cluster centre.
    (10) rbh - The (fiducial) radius of the central black hole.
    (11) delta_v - The systemic velocity of the system.

    In order to optimize any of the above parameters, the following
    observables must be provided for each star:
    (1) ra - The right ascension coordinate of the star.
    (2) dec - The declination coordinate of the star.
    (3) v - The measured radial velocity.
    (4) verr - The uncertainty of the measured velocity.
    """
    MODEL_PARAMETERS = ['d', 'mlr', 'barq', 'kappa_x', 'kappa_y', 'beta', 'mbh',
                        'ra_center', 'dec_center', 'rbh', 'delta_v']
    OBSERVABLES = {'ra': u.deg, 'dec': u.deg, 'v': u.km/u.s, 'verr': u.km/u.s}

    parameters_file = files(config).joinpath('axisymmetric.json')

    def __init__(self, data, parameters=None, mge_mass=None, mge_lum=None, mge_files=None, **kwargs):
        """
        Initialize a new instance of the Axisymmetric class.

        Parameters
        ----------
        data : instance of DataReader
            The measured positions and velocities of the trace population.
        parameters : instance of Parameters, optional
            The model parameters.
        mge_mass : instance of MgeReader, optional
            The Multi-Gaussian expansion (MGE) model of the projected mass
            density profile of the system (prior to any modifications from the
            mass-to-light ratio).
        mge_lum : instance of MgeReader, optional
            The Multi-Gaussian expansion (MGE) model of the projected
            luminosity density of the tracer population.
        mge_files : dict, optional
            A dictionary containing as keys various (x, y) offsets relative to
             the assumed centre (in arcsec!) and as values the filenames of
             different ECSV-files containing the MGE models calculated after
             shifting the centre by the provided offsets.
        kwargs
            Any additional keyword arguments are passed to the initialization
            of the parent Runner class.
        """
        if parameters is None:
            parameters = Parameters().load(self.parameters_file)

        # required observables
        self.ra = None
        self.dec = None

        super(Axisymmetric, self).__init__(data=data, parameters=parameters, **kwargs)

        assert isinstance(mge_mass, MgeReader) or mge_mass is None, "'mge_mass' must be instance of {0}".format(
            MgeReader.__module__)
        self.mge_mass = mge_mass

        assert isinstance(mge_lum, MgeReader) or mge_lum is None, "'mge_lum' must be instance of {0}".format(
            MgeReader.__module__)
        self.mge_lum = mge_lum
        
        if any([mge_mass is None, mge_lum is None]):
            assert all([mge_mass is None, mge_lum is None, mge_files is not None]), \
                "if 'mge_lum' is None or 'mge_mass' is None, both must be None and 'mge_files' must be given."
            assert mge_files is not None, "if 'mge_lum' or 'mge_mass' is None, you must provide 'mge_files'"
        
        self.use_mge_grid = mge_files is not None
        self.mge_files = mge_files

        # For
        if self.use_mge_grid:
            # we need a median_q value for the prior
            # for simplicity, we take the one from the central mge_profile of the grid
            idx = get_nearest_neigbhbour_idx2(0, 0, self.mge_files)
            _mge_lum, _ = get_mge(self.mge_files[idx])
            q_values = _mge_lum.data['q']
        else:
            q_values = self.mge_lum.data['q']
        self.median_q = np.median(q_values)
        self.min_q = np.min(q_values)

        # make sure limits on barq are considered in Parameters instance
        _ = find_barq_limits(q_values, parameters=self.parameters)

    def calculate_model_moments(self, values, return_model=False, **kwargs):

        current_parameters = self.fetch_parameter_values(values)
        
        unique_id = uuid.uuid4()

        with printoptions(precision=3):
            logstr = 'CJAM input parameters for {0}: '.format(unique_id)
            logstr += 'd={d:.2f}, mlr={mlr}, barq={barq:.3f}, kappa=[{kappa_x}, {kappa_y}]'.format(**current_parameters)
            logger.debug(logstr)

        # convert barq into inclination value
        if current_parameters['barq'] < 1:
            incl = np.arccos(np.sqrt(
                (self.median_q**2 - current_parameters['barq']**2)/(1. - current_parameters['barq']**2)))
        else:
            incl = 0.*u.rad

        # if we are using a MGE grid instead of a single MGE profile,
        # pick the MGE profile corresponding to the grid point closest to the offset
        if self.use_mge_grid:
            idx = get_nearest_neigbhbour_idx2(current_parameters['ra_center'].to(u.deg).value,
                                              current_parameters['dec_center'].to(u.deg).value,
                                              self.mge_files)
            mge_lum, mge_mass = get_mge(self.mge_files[idx])
            mge_lum, mge_mass = mge_lum.data, mge_mass.data
            
            gridpoint = mge_lum['gridpoint'].max()
            logger.debug("RA Center: {:.3f}, Dec Center: {:.3f}, gridpoint: {}".format(
                current_parameters['ra_center'], current_parameters['dec_center'], gridpoint))
            
        else:
            mge_lum = self.mge_lum.data
            mge_mass = self.mge_mass.data

        # get rotation amplitude
        kappa = np.sqrt(current_parameters['kappa_x']**2 + current_parameters['kappa_y']**2)

        # rotating data to determine rotation angle of cluster
        # copied from data_reader.DataReader.rotate()
        theta0 = np.arctan2(current_parameters['kappa_y'], current_parameters['kappa_x'])

        _x, _y = calc_xy_offset(ra=self.ra, dec=self.dec, ra_center=current_parameters['ra_center'],
                                dec_center=current_parameters['dec_center'])

        x = _x * np.cos(theta0) + _y * np.sin(theta0)
        y = -_x * np.sin(theta0) + _y * np.cos(theta0)
        
        # fixing cjam bug where it throws nans for star too close to centre
        xa = x.to(u.arcmin).value
        ya = y.to(u.arcmin).value
                
        xa = np.where((xa < 1e-3) & (xa > 0), 1e-3, xa)
        xa = np.where((xa > -1e-3) & (xa < 0), -1e-3, xa)
        
        ya = np.where((ya < 1e-3) & (ya > 0), 1e-3, ya)
        ya = np.where((ya > -1e-3) & (ya < 0), -1e-3, ya)        
        
        x = xa * u.arcmin
        y = ya * u.arcmin

        # calculate JAM model for current parameters
        try:
            model = cjam.axisymmetric(x, y, mge_lum, mge_mass, current_parameters['d'],
                                      beta=current_parameters['beta'], kappa=kappa, mscale=current_parameters['mlr'],
                                      incl=incl, mbh=current_parameters['mbh'], rbh=current_parameters['rbh'])

        except ValueError as err:
            logger.warning("CJAM returned an error:", err)
            return -np.inf

        logger.debug('CJAM call succeeded for {0}.'.format(unique_id))

        # get velocity and dispersion at every data point
        try:
            vz = model['vz']
            v2zz = model['v2zz']
        except TypeError:
            logger.warning('CJAM call failed.')
            return -np.inf
        v_los = vz - current_parameters['delta_v']

        # calculate likelihood
        if not (v2zz > vz**2).all():
            logging.error("Strange velocities or nan velocities for parameters: {}".format(current_parameters))
            return -np.inf

        if return_model:
            lnlike = self._calculate_lnlike(v_los=v_los, sigma_los=np.sqrt(v2zz - vz**2))
            return lnlike, x, y, vz, v2zz
        
        return self._calculate_lnlike(v_los=v_los, sigma_los=np.sqrt(v2zz - vz**2))

    def create_profiles(self, chain, n_burn, n_threads=1, n_samples=100, radii=None, n_theta=10,
                        filename=None, save_samples=False):
        """
        Create radial profiles of the (projected) rotation velocity and the
        velocity dispersion by randomly drawing parameters samples from the
        provided chain, recovering the corresponding velocity moments using
        cjam, and averaging the final set of velocity moments.

        Parameters
        ----------
        chain : nd_array
            The chain containing the parameter samples should have shape
            (n_parameters, n_steps, n_walkers).
        n_burn : int
            The number of steps to be disregarded at the beginning of each
            walker.
        n_threads : int, optional
            The number of parallel threads to be used for recovering the
            velocity moments using cjam.
        n_samples : int, optional
            The number of parameter samples to be drawn from the chain.
        radii : ndarray, optional
            The radii at which the profiles should be sampled.
        n_theta : int, optional
            The number of steps in position angle space.
        filename : str, optional
            The name of the file used to store the final profiles,
            in ecsv-format.
        save_samples : bool, optional

        Returns
        -------
        profile : an astropy QTable
            The radial profiles for the rotation velocity and the velocity
            dispersion. For each quantity, the median and the 1- and 3-sigma
            intervals of the individual curves are returned.
        """
        # get positions where to sample model
        if radii is None:
            radii = np.logspace(-1, 3, 200)*u.arcsec
        theta = np.linspace(0, 2. * np.pi, n_theta, endpoint=False)*u.rad

        x = (radii[:, np.newaxis] * np.cos(theta)).flatten()
        y = (radii[:, np.newaxis] * np.sin(theta)).flatten()

        # get parameter sets
        parameters = self.sample_chain(chain=chain, n_burn=n_burn, n_samples=n_samples)

        # replace barq with inclination in parameter sets
        for i in range(len(parameters)):
            barq = parameters[i].pop('barq')
            parameters[i]['incl'] = np.arccos(np.sqrt((self.median_q**2 - barq**2)/(1. - barq**2)))

            if 'kappa_x' in parameters[i] and 'kappa_y' in parameters[i] and 'kappa' not in parameters[i]:
                parameters[i]['kappa'] = np.sqrt(parameters[i]['kappa_x']**2 + parameters[i]['kappa_y']**2)

        if self.use_mge_grid:
            for i, p in enumerate(parameters):
                idx = get_nearest_neigbhbour_idx2(p['ra_center'].to(u.deg).value, p['dec_center'].to(u.deg).value,
                                                  self.mge_files)
                parameters[i]['mge_filename'] = self.mge_files[idx]
        else:
            for i, p in enumerate(parameters):
                parameters[i]['mge_filename'] = None 
                
        # run cjam for selected parameter sets
        logger.info('Recovering models using {0} threads ...'.format(n_threads))
        
        if self.use_mge_grid:
            init_arguments = (x, y, None, None)
        else:
            init_arguments = (x, y, self.mge_mass.data, self.mge_lum.data)
            
        if n_threads > 1:
            pool = Pool(n_threads, initializer=init_cjam, initargs=init_arguments)
            _results = pool.map_async(run_cjam, parameters)
            results = _results.get()
        else:
            init_cjam(*init_arguments)
            results = [run_cjam(p) for p in parameters]

        good_results = [r for r in results if np.isfinite(r).all()]
        results = good_results
        
        # get percentiles of mean velocity and dispersion
        vz = np.percentile([r[0] for r in results], [50, 16, 84, 0.15, 99.85], axis=0)
        sigma = np.percentile([np.sqrt(r[1] - r[0]**2) for r in results], [50, 16, 84, 0.15, 99.85], axis=0)

        # for rotation velocity, use 1st order moments along positive x-axis (=semi-major axis)
        semimajor = np.mod(np.arange(x.size), theta.size) == 0
        vz_radial = vz[:, semimajor]*u.km/u.s

        # for dispersion, average 2nd order moment over data points with same radius
        sameradius = np.arange(x.size) // theta.size
        sigma_radial = [stats.binned_statistic(sameradius, s, 'mean', bins=radii.size)[0] for s in sigma]*u.km/u.s

        # store radial profiles astropy table
        profile = Table(
            [Table.Column(radii, name='r', dtype=np.float64, unit=radii.unit),
             Table.Column(vz_radial[0], name='v_rot', unit=vz_radial.unit),
             Table.Column(vz_radial[1], name='v_rot_lower_1s', unit=vz_radial.unit),
             Table.Column(vz_radial[2], name='v_rot_upper_1s', unit=vz_radial.unit),
             Table.Column(vz_radial[3], name='v_rot_lower_3s', unit=vz_radial.unit),
             Table.Column(vz_radial[4], name='v_rot_upper_3s', unit=vz_radial.unit),
             Table.Column(sigma_radial[0], name='sigma', unit=sigma_radial.unit),
             Table.Column(sigma_radial[1], name='sigma_lower_1s', unit=sigma_radial.unit),
             Table.Column(sigma_radial[2], name='sigma_upper_1s', unit=sigma_radial.unit),
             Table.Column(sigma_radial[3], name='sigma_lower_3s', unit=sigma_radial.unit),
             Table.Column(sigma_radial[4], name='sigma_upper_3s', unit=sigma_radial.unit)])

        if filename is not None:
            profile.write(filename, format='ascii.ecsv', overwrite=True)
            
        if save_samples:
            # parameters is a list of dicts, we need a dict with lists
            # stolen from stackoverflow
            # v = {k: [dic[k] for dic inparameters] for k in parameters[0]}
            
            allsamples = []
            for i, param in enumerate(parameters):
                samples = pd.DataFrame({'x': x, 'y': y, 'first_moment': results[i][0], 'second_moment': results[i][1]})
                for k, v in param.items():
                    samples[k] = v
                allsamples.append(samples)
                
            allsamples = pd.concat(allsamples, ignore_index=True)
            fname = filename[:filename.find('.')] + '_allsamples.csv'
            allsamples.to_csv(fname, index=False)

        return profile

    def calculate_mlr_profile(self, mlr, radii=None, mge_mass=None):
        """
        The method calculates a radial profile of the mass-to-light ratio.

        Parameters
        ----------
        mlr : array_like
            The scaling factors of the individual Gaussian components of the
            MGE representing the mass surface density of the cluster. The
            length must match the number of MGE components.
        radii : astropy.unit.Quantity
            The radii at which the mass-to-light ratio should be calculated.
            If none are provided, a sequence of log-sampled radii is created
            internally.
        mge_mass: files.MgeReader
            If given, use this MGE instead of self.mge_mass. Useful when using 
            a MGE grid.

        Returns
        -------
        r : astropy.unit.Quantity
            The radii at which the mass-to-light ratio were calculated.
        mlr_profile : array_like
            The mass-to-light ratio of the cluster as a function of distance
            to the cluster centre.
        """
        
        _mge_mass = self.mge_mass if mge_mass is None else mge_mass
        
        if mge_mass is not None and radii is None:
            logger.warning(
                "No radii given but explicit MGE is used. Automatically set radii will change with MGEs!")
        
        if radii is None:
            rmin = _mge_mass.data['s'].min().value
            rmax = _mge_mass.data['s'].max().value
            radii = np.logspace(np.log10(rmin) - 0.5, np.log10(rmax) + 0.5, 50) * _mge_mass.data['s'].unit

        radii = u.Quantity(radii)
        if radii.unit.is_unity():
            radii *= _mge_mass.data['s'].unit
            logger.warning('Cannot determine unit for parameter <r>. Assuming {0}.'.format(radii.unit))

        assert len(mlr) == len(_mge_mass.data), "Length of parameter <mlr> must match no. of MGE components."
        mlr = u.Quantity(mlr)

        mlr_profile = np.zeros((radii.size,), dtype=np.float64)*_mge_mass.data['i'].unit*mlr.unit
        total = np.zeros_like(mlr_profile)/mlr.unit
        for j, row in enumerate(_mge_mass.data):
            gaussian = row['i']*np.exp(-0.5 * (radii / (np.sqrt(1. - row['q']) * row['s'])) ** 2)

            total += gaussian
            mlr_profile += mlr[j]*gaussian

        return radii, mlr_profile/total
