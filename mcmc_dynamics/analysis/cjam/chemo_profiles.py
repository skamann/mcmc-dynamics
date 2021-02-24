import logging
import uuid
import numpy as np
import cjam
from multiprocessing import Pool
from string import ascii_lowercase
from scipy import stats
from astropy import units as u
from astropy.table import Table
from .axisymmetric import printoptions
from .radial_profiles import AnalyticalProfiles


logger = logging.getLogger(__name__)


def init_cjam(x, y, mge_mass, *args):

    global gx
    global gy
    global gmge_mass

    gx = x
    gy = y
    gmge_mass = mge_mass


def run_cjam(parameters):

    global gx, gy, gmge_mass

    model = cjam.axisymmetric(gx, gy, parameters['mge_lum'], gmge_mass, parameters['d'], beta=parameters['beta'],
                              kappa=parameters['kappa'], mscale=parameters['mlr'].value, incl=parameters['incl'])

    # get velocity and dispersion at every data point
    # note that astropy Quanitities cannot be pickled, so multiprocessing only works when the values are returned
    return model['vz'].value, model['v2zz'].value


class ChemoProfiles(AnalyticalProfiles):

    def __init__(self, data, mge_mass, mge_lum, initials, n_pops, **kwargs):

        # required observables
        self.feh = None
        self.feherr = None

        self.n_pops = n_pops

        super(ChemoProfiles, self).__init__(data=data, mge_mass=mge_mass, mge_lum=mge_lum, initials=initials, **kwargs)

        # check if luminosity MGE has population labels
        if 'pop' not in self.mge_lum.data.columns:
            logger.error('Missing population tags (column "pop") in luminosity MGE.')
        if not np.array_equal(np.unique(self.mge_lum.data['pop']), np.arange(self.n_pops)):
            logger.error('Population tags in luminosity MGE inconsistent with {0} populations.'.format(self.n_pops))

        # get radii where to assign mlr, kappa to Gaussian components
        self.x_mlr = self.find_mge_peaks(self.mge_mass.data['s'], self.mge_mass.data['i'])
        self.x_mlr[self.mge_mass.data['s'].argmin()] = 0.
        self.x_mlr[self.mge_mass.data['s'].argmax()] *= 10

        self.x_kappa = np.zeros(self.mge_lum.n_components)*self.mge_lum.data['s'].unit
        for n in range(self.n_pops):
            slc = self.mge_lum.data['pop'] == n
            _x_kappa = self.find_mge_peaks(self.mge_lum.data['s'][slc], self.mge_lum.data['i'][slc])
            _x_kappa[self.mge_lum.data['s'][slc].argmin()] = 0.
            _x_kappa[self.mge_lum.data['s'][slc].argmax()] *= 10
            self.x_kappa[slc] = _x_kappa

        p_spatial = np.zeros((self.n_pops, self.data.sample_size))
        for n in range(self.n_pops):
            slc = self.mge_lum.data['pop'] == n
            p_spatial[n] = self.mge_lum.eval(self.x, self.y, n=self.mge_lum.data['n'][slc])
        self.p_spatial = p_spatial/self.mge_lum.eval(self.x, self.y)

    @property
    def observables(self):
        if self._observables is None:
            self._observables = super(ChemoProfiles, self).observables
            self._observables.update({'feh': u.dimensionless_unscaled, 'feherr': u.dimensionless_unscaled})
        return self._observables

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(ChemoProfiles, self).parameters
            _ = self._parameters.pop('kappa_max')
            _ = self._parameters.pop('r_kappa')
            for i in range(self.n_pops):
                self._parameters['mu_feh_{0}'.format(ascii_lowercase[i])] = u.dimensionless_unscaled
                self._parameters['sigma_feh_{0}'.format(ascii_lowercase[i])] = u.dimensionless_unscaled
                self._parameters['kappa_max_{0}'.format(ascii_lowercase[i])] = u.dimensionless_unscaled
                self._parameters['logr_kappa_{0}'.format(ascii_lowercase[i])] = u.dex(u.arcmin)
                # if i < (self.n_pops - 1):
                #     self._parameters['f_{0}'.format(ascii_lowercase[i])] = u.dimensionless_unscaled
                for j in range(self.n_pops - 1):
                    self._parameters[
                        'h_{0}{1}'.format(ascii_lowercase[j], ascii_lowercase[i])] = u.dimensionless_unscaled
        return self._parameters

    @property
    def parameter_labels(self):
        labels = {}
        for row in self.initials:
            name = row['name']
            latex_string = row['init'].unit.to_string('latex')
            if name == 'd':
                labels[name] = r'$d/${0}'.format(latex_string)
            elif len(name) > 3 and name[:3] == 'mlr':
                suffix = name[4:]
                if suffix == 'inf':
                    suffix = r'\infty'
                labels[name] = r'$\Upsilon_{{\rm {0}}}/{{\rm M_\odot}}\,{{\rm L_\odot^{{-1}}}}$'.format(suffix)
            elif name == 'barq':
                labels[name] = r'$\bar{q}$'
            elif len(name) > 10 and name[:10] == 'kappa_max_':
                suffix = name[-1].upper()
                labels[name] = r'$\kappa_{{\rm max.,\,{0}}}$'.format(suffix)
            elif len(name) > 7 and name[:7] == 'mu_feh_':
                suffix = name[-1].upper()
                labels[name] = r'$\mu_{{\rm chem.,\,{0}}}$'.format(suffix)
            elif len(name) > 10 and name[:10] == 'sigma_feh_':
                suffix = name[-1].upper()
                labels[name] = r'$\sigma_{{\rm chem.,\,{0}}}$'.format(suffix)
            elif len(name) > 2 and name[:2] == 'h_':
                suffix_0 = name[-2].upper()
                suffix_1 = name[-1].upper()
                labels[name] = r'$h_{{\rm {0}{1}}}$'.format(suffix_0, suffix_1)
            elif name == 'beta':
                labels[name] = r'$\beta$'
            elif len(name) > 10 and name[:10] == 'logr_kappa':
                suffix = name[-1].upper()
                labels[name] = r'$\log(r_{{\rm \kappa,\,{0}}}$/{1})'.format(suffix, latex_string)
            elif name == 'r_mlr':
                labels[name] = r'$r_{{\rm \Upsilon}}$/{0}'.format(latex_string)
            else:
                labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)

        return labels

    @staticmethod
    def find_mge_peaks(sigma, intensity):
        # MGE components are assigned the values of the analytical functions at the distances where their contribution
        # to the overall profiles is max.
        x = np.logspace(u.Dex(sigma).min().value, u.Dex(sigma).max().value, 100)*sigma.unit
        weights = np.zeros((x.size, len(sigma)))
        for i in range(weights.shape[1]):
            weights[:, i] = intensity[i] * np.exp(-0.5 * (x / sigma[i]) ** 2)
        weights /= weights.sum(axis=1)[:, np.newaxis]
        return x[weights.argmax(axis=0)]

    def fetch_parameter_values(self, values):

        parameters = super(AnalyticalProfiles, self).fetch_parameter_values(values)

        _x = self.x_mlr/parameters.pop('r_mlr')
        parameters['mlr'] = (
            parameters.pop('mlr_0')*(1.-_x) + 2.*parameters.pop('mlr_t')*_x + parameters.pop('mlr_inf')*_x*(_x-1.))/(
                1.+_x**2)

        for i in range(self.n_pops):
            logr_kappa = parameters.pop('logr_kappa_{0}'.format(ascii_lowercase[i]))
            _x = self.x_kappa / logr_kappa.physical
            kappa_max = parameters.pop('kappa_max_{0}'.format(ascii_lowercase[i]))
            parameters['kappa_{0}'.format(ascii_lowercase[i])] = 2.*kappa_max*_x / (1. + _x**2)

        pop_max = ascii_lowercase[self.n_pops - 1]
        # parameters['f_{0}'.format(pop_max)] = 1. - np.sum(
        #     [parameters['f_{0}'.format(ascii_lowercase[i])] for i in range(self.n_pops - 1)])
        for i in range(self.n_pops):
            parameters['h_{0}{1}'.format(pop_max, ascii_lowercase[i])] = 1. - np.sum(
                [parameters['h_{0}{1}'.format(ascii_lowercase[j], ascii_lowercase[i])] for j in range(self.n_pops - 1)])

        return parameters

    def lnprior(self, values):

        parameters = self.fetch_parameter_values(values)

        for parameter, value in parameters.items():

            if len(parameter) == 4 and parameter[:2] == 'h_' and not (0 <= value <= 1):
                return -np.inf
            # elif len(parameter) == 3 and parameter[:2] == 'f_' and value <= 0:
            #     return -np.inf
            elif len(parameter) == 8 and parameter[:7] == 'mu_feh_':
                if not (-1 < value < 1):
                    return -np.inf
                # make sure the populations "do not cross"
                i = ascii_lowercase.index(parameter[-1])
                if 'mu_feh_{0}'.format(ascii_lowercase[i-1]) in parameters and value <= parameters[
                        'mu_feh_{0}'.format(ascii_lowercase[i-1])]:
                    return -np.inf
                elif 'mu_feh_{0}'.format(ascii_lowercase[i+1]) in parameters and value >= parameters[
                        'mu_feh_{0}'.format(ascii_lowercase[i+1])]:
                    return -np.inf
            elif len(parameter) == 11 and parameter[:10] == 'sigma_feh_' and value < 0:
                return -np.inf

        return super(ChemoProfiles, self).lnprior(values=values)

    def lnlike(self, values, individual=False):

        current_parameters = self.fetch_parameter_values(values)

        # convert barq into inclination value
        incl = np.arccos(np.sqrt(
            (self.median_q**2 - current_parameters['barq']**2)/(1. - current_parameters['barq']**2)))

        lnlike = []

        # loop over populations
        for i in range(self.n_pops):
            pop = ascii_lowercase[i]

            h_values = [current_parameters['h_{0}{1}'.format(pop, ascii_lowercase[j])] for j in range(self.n_pops)]

            mge_lum = self.mge_lum.data.copy()
            for j in range(self.n_pops):
                mge_lum['i'][mge_lum['pop'] == j] *= h_values[j]
            kappa = current_parameters['kappa_{0}'.format(pop)]

            if (mge_lum['i'] == 0).any():
                has_signal = mge_lum['i'] != 0
                mge_lum = mge_lum[has_signal]
                kappa = kappa[has_signal]

            unique_id = uuid.uuid4()

            with printoptions(precision=3):
                print_parameters = current_parameters.copy()
                print_parameters['kappa_{0}'.format(pop)] = kappa

                logstr = 'CJAM input parameters for {0}: '.format(unique_id)
                logstr += 'd={{d:.2f}}, mlr={{mlr}}, barq={{barq:.3f}}, kappa={{kappa_{0}}}'.format(pop)
                logger.debug(logstr.format(**print_parameters))

            # calculate JAM model for current parameters
            model = cjam.axisymmetric(self.x, self.y, mge_lum, self.mge_mass.data, current_parameters['d'],
                                      beta=current_parameters['beta'], kappa=kappa,
                                      mscale=current_parameters['mlr'], incl=incl)

            logger.debug('CJAM call succeeded for {0}.'.format(unique_id))

            # get velocity and dispersion at every data point
            vz = model['vz']
            v2zz = model['v2zz']

            if not (v2zz > vz**2).all():
                return -np.inf

            v_los = vz
            sigma_los = np.sqrt(v2zz - vz**2)

            like_spatial = np.sum([self.p_spatial[j]*h_values[j] for j in range(self.n_pops)], axis=0)

            norm_v = self.verr * self.verr + sigma_los * sigma_los
            exponent_v = -0.5 * np.power(self.v - v_los, 2) / norm_v

            norm_feh = self.feherr * self.feherr + current_parameters['sigma_feh_{0}'.format(pop)]**2
            exponent_feh = -0.5 * np.power(self.feh - current_parameters['mu_feh_{0}'.format(pop)], 2) / norm_feh

            lnlike.append(np.log(like_spatial) - 0.5 * np.log(2. * np.pi * norm_v.value) + exponent_v - 0.5 * np.log(
                2. * np.pi * norm_feh.value) + exponent_feh)

        if individual:
            return lnlike
        else:
            max_lnlike = np.max(lnlike, axis=0)
            like = np.sum([np.exp(lnlike[i] - max_lnlike) for i in range(self.n_pops)], axis=0)
            return np.sum(max_lnlike + np.log(like))

    def get_initials(self, n_walkers):
        initials = np.zeros((n_walkers, self.n_fitted_parameters))
        i = 0
        for row in self.initials:
            if row['fixed']:
                continue
            elif row['name'] == 'barq':
                initials[:, i] = self.median_q - 0.1*np.random.rand(n_walkers)
            elif len(row['name']) >= 5 and row['name'][:5] == 'kappa':
                initials[:, i] = row['init'] + 0.3*np.random.randn(n_walkers)
            elif len(row['name']) >= 10 and row['name'][:10] == 'logr_kappa':
                initials[:, i] = row['init'].value + 0.2 * np.random.randn(n_walkers)
            elif len(row['name']) >= 6 and row['name'][:6] == 'mu_feh':
                initials[:, i] = row['init'] + 0.08*np.random.rand(n_walkers) - 0.04
            elif len(row['name']) >= 9 and row['name'][:9] == 'sigma_feh':
                initials[:, i] = row['init'] * (1. + 0.3 * np.random.randn(n_walkers))
            # elif len(row['name']) == 3 and row['name'][:2] == 'f_':
            #     initials[:, i] = np.random.rand(n_walkers)/(self.n_pops - 1.)
            elif len(row['name']) == 4 and row['name'][:2] == 'h_':
                if row['name'][2] == row['name'][3]:
                    initials[:, i] = 0.8 + 0.1*np.random.rand(n_walkers)
                else:
                    initials[:, i] = 0.1*np.random.rand(n_walkers)/(self.n_pops - 2.)
            else:
                initials[:, i] = row['init'] * (0.7 + 0.6*np.random.rand(n_walkers))*row['init'].unit
            i += 1
        return initials

    def create_profiles(self, chain, n_burn, n_threads=1, n_samples=100, radii=None, n_theta=10,
                        prefix=None):
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
        prefix : str, optional
            The common prefix of the names of the files used to store the
            final profiles, in ecsv-format. One file per population will be
            created.

        Returns
        -------
        profiles : dict
            The dictionary will contain one astropy Table per population. Each
            table will hold the radial profiles for the rotation velocity and
            the velocity dispersion. For each quantity, the median and the 1-
            and 3-sigma intervals of the individual curves are returned.
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
        for n in range(len(parameters)):
            barq = parameters[n].pop('barq')
            parameters[n]['incl'] = np.arccos(np.sqrt((self.median_q**2 - barq**2)/(1. - barq**2)))

        profiles = {}

        # loop over populations
        for i in range(self.n_pops):
            pop = ascii_lowercase[i]

            for n in range(len(parameters)):
                h_values = [parameters[n]['h_{0}{1}'.format(pop, ascii_lowercase[j])] for j in range(self.n_pops)]

                mge_lum = self.mge_lum.data.copy()
                for j in range(self.n_pops):
                    mge_lum['i'][mge_lum['pop'] == j] *= h_values[j]

                kappa = parameters[n]['kappa_{0}'.format(pop)]

                if (mge_lum['i'] == 0).any():
                    has_signal = mge_lum['i'] != 0
                    mge_lum = mge_lum[has_signal]
                    kappa = kappa[has_signal]

                parameters[n]['mge_lum'] = mge_lum
                parameters[n]['kappa'] = kappa

            # run cjam for selected parameter sets
            logger.info('Recovering models using {0} threads ...'.format(n_threads))
            init_arguments = (x, y, self.mge_mass.data)
            if n_threads > 1:
                pool = Pool(n_threads, initializer=init_cjam, initargs=init_arguments)
                _results = pool.map_async(run_cjam, parameters)
                results = _results.get()
            else:
                init_cjam(*init_arguments)
                results = [run_cjam(p) for p in parameters]

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
            profiles[i] = Table(
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

            if prefix is not None:
                profiles[i].write('{0}_pop{1}.ecsv'.format(prefix, i), format='ascii.ecsv', overwrite=True)

        return profiles

