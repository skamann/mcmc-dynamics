import contextlib
import logging
import uuid
import numpy as np
import cjam
from multiprocessing import Pool
from scipy import stats
from astropy import units as u
from astropy.table import Table
from ..runner import Runner
from mcmc_dynamics.utils.files import MgeReader, get_mge, get_nearest_neigbhbour_idx2


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

    global gx
    global gy
    global gmge_mass
    global gmge_lum

    gx = x
    gy = y
    gmge_mass = mge_mass
    gmge_lum = mge_lum


def run_cjam(parameters):

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
    # note that astropy Quanitities cannot be pickled, so multiprocessing only works when the values are returned
    return model['vz'].value, model['v2zz'].value


class Axisymmetric(Runner):

    def __init__(self, data, initials, mge_mass=None, mge_lum=None, mge_coords=None, mge_files=None, **kwargs):
        # required observables
        self.x = None
        self.y = None

        super(Axisymmetric, self).__init__(data=data, initials=initials, **kwargs)

        # # Get required columns. If units are available, make sure they are as we expect.
        # # IMPORTANT: It is assumed that x is aligned with the (projected) semi-major axis of the system!
        # self.x = u.Quantity(data.data['x'])
        # if self.x.unit.is_unity():
        #     self.x *= u.arcsec
        #     logger.warning('Missing unit for <x> values. Assuming {0}.'.format(self.x.unit))
        # self.y = u.Quantity(data.data['y'])
        # if self.y.unit.is_unity():
        #     self.y *= u.arcsec
        #     logger.warning('Missing unit for <y> values. Assuming {0}.'.format(self.y.unit))

        assert isinstance(mge_mass, MgeReader) or mge_mass is None, "'mge_mass' must be instance of {0}".format(MgeReader.__module__)
        self.mge_mass = mge_mass

        assert isinstance(mge_lum, MgeReader) or mge_lum is None, "'mge_lum' must be instance of {0}".format(MgeReader.__module__)
        self.mge_lum = mge_lum
        
        if any([mge_mass is None, mge_lum is None]):
            assert all([mge_mass is None, mge_lum is None, mge_files is not None]), "if mge_lum is None or mge_mass is None, both must be None and mge_files must be given."
            assert mge_files is not None, "if mge_lum or mge_mass is None, you must provide mge_files"
        
        self.use_mge_grid = mge_files is not None
        self.mge_files = mge_files

        if self.use_mge_grid:
            # we need a median_q value for the prior
            # for simplicity, we take the one from the central mge_profile of the grid
            idx = get_nearest_neigbhbour_idx2(0, 0, self.mge_files)
            _mge_lum, _ = get_mge(self.mge_files[idx])
            self.median_q = np.median(_mge_lum.data['q'])
        else:
            self.median_q = np.median(self.mge_lum.data['q'])
            

    @property
    def observables(self):
        if self._observables is None:
            self._observables = super(Axisymmetric, self).observables
            self._observables.update({'x': u.arcsec, 'y': u.arcsec})
        return self._observables

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(Axisymmetric, self).parameters
            self._parameters.update({'d': u.kpc, 'mlr': u.dimensionless_unscaled, 'barq': u.dimensionless_unscaled,
                                     'kappa_x': u.dimensionless_unscaled, 'kappa_y': u.dimensionless_unscaled,
                                     'beta': u.dimensionless_unscaled, 'mbh': u.Msun, 
                                     'delta_x': u.arcsec, 'delta_y': u.arcsec, 'rbh': u.arcsec,
                                     'delta_v': u.km/u.s})
        return self._parameters

    @property
    def parameter_labels(self):
        labels = {}
        for row in self.initials:
            latex_string = row['init'].unit.to_string('latex')
            if row['name'] == 'd':
                labels[row['name']] = r'$d/${0}'.format(latex_string)
            elif row['name'] == 'mlr':
                labels[row['name']] = r'$\Upsilon/\frac{\rm M_\odot}{\rm L_\odot}$'
            elif row['name'] == 'barq':
                labels[row['name']] = r'$\bar{q}$'
            elif row['name'] == 'kappa':
                labels[row['name']] = r'$\kappa$'
            elif row['name'] == 'beta':
                labels[row['name']] = r'$\beta$'
            elif row['name'] == 'delta_v':
                labels[row['name']] = r'$\Delta v$'
#            elif row['name'] == 'theta_0':
#                labels[row['name']] = r'$\theta_{{\rm 0}}/${0}'.format(latex_string)
            else:
                labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
        return labels

    def lnprior(self, values):
        p = 0
        current_parameters = self.fetch_parameters(values)
        
        for parameter, value in current_parameters.items():
            if parameter == 'd' and value <= 0.5*u.kpc:
                return -np.inf
            elif parameter == 'mlr' and (np.less_equal(value, 0.1).any() or np.greater(value, 10).any()):
                return -np.inf
            elif parameter == 'barq' and (value <= 0.2 or value > self.median_q):
                return -np.inf
            elif parameter == 'kappa_x' or parameter =='kappa_y':
                p += stats.norm.logpdf(value, 0, 5)
            elif parameter == 'delta_x' or parameter == 'delta_y':
                p += stats.norm.logpdf(value, 0, 1)
            elif parameter == 'delta_v':
                p += stats.norm.logpdf(value, 0, 1)
            elif parameter == 'mbh':
                p += stats.uniform.logpdf(value, 0, 15000)

        return p + super(Axisymmetric, self).lnprior(values=values)

    def lnlike(self, values, return_model=False):
        x = np.copy(self.x)
        y = np.copy(self.y)
        
        current_parameters = self.fetch_parameters(values)
        
        unique_id = uuid.uuid4()

        with printoptions(precision=3):
            logstr = 'CJAM input parameters for {0}: '.format(unique_id)
            logstr += 'd={d:.2f}, mlr={mlr}, barq={barq:.3f}, kappa={kappa}'.format(**current_parameters)
            logger.debug(logstr)

        # convert barq into inclination value
        incl = np.arccos(np.sqrt(
            (self.median_q**2 - current_parameters['barq']**2)/(1. - current_parameters['barq']**2)))

        x -= current_parameters['delta_x']
        y -= current_parameters['delta_y']
        
        # if we are using a MGE grid instead of a single MGE profile,
        # pick the MGE profile corresponding to the grid point closes to the offset
        if self.use_mge_grid:
            idx = get_nearest_neigbhbour_idx2(-current_parameters['delta_x'].to(u.arcsec).value, 
                                              -current_parameters['delta_y'].to(u.arcsec).value, 
                                              self.mge_files)
            mge_lum, mge_mass = get_mge(self.mge_files[idx])
            mge_lum, mge_mass = mge_lum.data, mge_mass.data
            
            gridpoint = mge_lum['gridpoint'].max()
            logger.info("delta_x: {:.3f}, delta_y: {:.3f}, gridpoint: {}".format(current_parameters['delta_x'], current_parameters['delta_y'], gridpoint))
            
        else:
            mge_lum = self.mge_lum.data
            mge_mass = self.mge_mass.data
        
        # rotating data to determine rotation angle of cluster
        # copied from data_reader.DataReader.rotate()
        theta0 = np.arctan2(current_parameters['kappa_y'], current_parameters['kappa_x'])
        
        xnew = x * np.cos(theta0) + y * np.sin(theta0)
        ynew = -x * np.sin(theta0) + y * np.cos(theta0)
        
        x = xnew
        y = ynew        

        # fixing cjam bug where it throws nans for star too close to centre

        xa = x.to(u.arcmin).value
        ya = y.to(u.arcmin).value
                
        xa = np.where((xa < 1e-3) & (xa > 0), 1e-3, xa)
        xa = np.where((xa > -1e-3) & (xa < 0), -1e-3, xa)
        
        ya = np.where((ya < 1e-3) & (ya > 0), 1e-3, ya)
        ya = np.where((ya > -1e-3) & (ya < 0), -1e-3, ya)        
        
        x = xa * u.arcmin
        y = ya * u.arcmin
        
        self.v += current_parameters['delta_v']

        # calculate JAM model for current parameters
        try:
            model = cjam.axisymmetric(x, y, mge_lum, mge_mass, current_parameters['d'],
                                      beta=current_parameters['beta'], kappa=current_parameters['kappa'],
                                      mscale=current_parameters['mlr'], incl=incl, mbh=current_parameters['mbh'],
                                      rbh=current_parameters['rbh'])

        except ValueError as err:
            logging.warn("CJAM returned an error:", err)
            return -np.inf

        logger.debug('CJAM call succeeded for {0}.'.format(unique_id))

        # get velocity and dispersion at every data point
        vz = model['vz']
        v2zz = model['v2zz']

        # calculate likelihood
        if not (v2zz > vz**2).all():
            logging.error("Strange velocities or nan velocities for parameters: {}".format(current_parameters))
            return -np.inf

        if return_model:
            lnlike = self._calculate_lnlike(v_los=vz, sigma_los=np.sqrt(v2zz - vz**2))
            return lnlike, x, y, vz, v2zz
        
        return self._calculate_lnlike(v_los=vz, sigma_los=np.sqrt(v2zz - vz**2))

    def get_initials(self, n_walkers):
        initials = np.zeros((n_walkers, self.n_fitted_parameters))
        i = 0
        for row in self.initials:
            if row['fixed']:
                continue
            elif row['name'] == 'barq':
                initials[:, i] = self.median_q - 0.5*np.random.rand(n_walkers)
            elif row['name'] == 'kappa_x' or row['name'] == 'kappa_y':
                initials[:, i] = 2*row['init']*np.random.rand(n_walkers) - row['init']
            elif len(row['name']) >= 5 and row['name'][:5] == 'kappa':
                initials[:, i] = row['init'] + 0.3*np.random.randn(n_walkers)
            elif row['name'] == 'mbh':
                initials[:, i] = np.random.rand(n_walkers) * row['init']
            elif row['name'] == 'delta_x' or row['name'] == 'delta_y':
                # uniform on [-init, +init]
                initials[:, i] = 2*row['init']*np.random.rand(n_walkers) - row['init']
            elif row['name'] == 'delta_v':
                initials[:, i] = 2*row['init']*np.random.rand(n_walkers) - row['init']
            elif row['name'] == 'r_kappa':
                a = 10
                b= 150
                initials[:, i] = (b-a) * np.random.rand(n_walkers) + a
            elif row['name'] == 'r_mlr':
                a = 10
                b = 150
                initials[:, i] = (b-a) * np.random.rand(n_walkers) + a
              
            else:
                initials[:, i] = row['init'] * (0.7 + 0.6*np.random.rand(n_walkers))*row['init'].unit
            i += 1
            
        return initials

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
            
        if self.use_mge_grid:
            for i, p in enumerate(parameters):
                idx = get_nearest_neigbhbour_idx2(-p['delta_x'].to(u.arcsec).value, -p['delta_y'].to(u.arcsec).value, self.mge_files)
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
            #v = {k: [dic[k] for dic inparameters] for k in parameters[0]}
            
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
            logger.warning("No radii given but explicit MGE is used. The automatic radii used will be different for different MGEs!")
        
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

