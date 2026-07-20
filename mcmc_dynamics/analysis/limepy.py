import inspect
import logging
try:
    from importlib.resources import files
except ImportError:  # for Python v<3.9
    from importlib_resources import files
import numpy as np
from scipy import interpolate
from astropy import units as u
from astropy.table import  Table
from limepy import limepy

from .runner import Runner
from .. import config
from ..parameter import Parameters
from ..utils.coordinates import calc_xy_offset

logger = logging.getLogger(__name__)


class LimepyFit(Runner):
    """
    The purpose of the Limepy class is to fit the radial dispersion profile of
    a cluster using a Limepy model. Note that Limepy models are non-rotating.

    For a detailed description of Limepy models, see `Gieles & Zocchi 2015`_.

    The data required per star are the cartesian coordinates :math:`x` and
    :math:`y` to the cluster centre, the radial velocity :math:`v` and the
    velocity uncertainty :math:`\\epsilon_v`.

    References
    ----------
    .. _Gieles & Zocchi 2015:
       https://ui.adsabs.harvard.edu/abs/2015MNRAS.454..576G/abstract

    Parameters
    ----------
    data : instance of DataReader
        The observed data for a set of n stars. The instance must provide
        at least the radii, the position angles, the velocities, and their
        uncertainties.
    parameters : instance of Parameters, optional
        The model parameters.
    kwargs :
        Any additional keyword arguments are passed on to the
        initialization of the parent class.
    """
    MODEL_PARAMETERS = ['v_sys', 'phi0', 'g', 'rh', 'mass', 'ra_center', 'dec_center']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'ra': u.deg, 'dec': u.deg}

    parameters_file = files(config).joinpath('limepy.json')

    def __init__(self, data, parameters=None, distance=10.*u.kpc, **kwargs):
        """
        Initialize a new instance of the ModelFit class
        """
        # required observables
        self.ra = None
        self.dec = None

        self.pc2arcmin = (1.*u.arcmin/distance.to(u.pc))*(360.*60./(2.*np.pi))

        if parameters is None:
            parameters = Parameters().load(self.parameters_file)

        super(LimepyFit, self).__init__(data=data, parameters=parameters, **kwargs)

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters
        self.dispersion_parameters = inspect.signature(self.dispersion_model).parameters

    def dispersion_model(self, phi0, g, ra_center, dec_center, rh=3, mass=1e5, **kwargs):
        """
        The method calculates the line-of-sight velocity dispersion at the
        positions (r, theta) of the available data points.

        Parameters
        ----------
        phi0 : float
            The central dimensionless potential
        g : float
            Order of truncation (0<= g < 3.5; 0=Woolley, 1=King, 2=Wilson)
        ra_center : float
            The right ascension of the model centre.
        dec_center : float
            The declination of the model centre.
        rh : float
            The scale radius of the model.
        mass : float
            The final scaled mass of the model.
        kwargs
            This method does not use any additional keyword arguments.

        Returns
        -------
        sigma_los : ndarray
            The line-of-sight velocity dispersion of the model evaluated at
            the positions of the individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.dispersion_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        lpmodel = limepy(phi0=phi0.value, g=g.value, rh=rh.value, M=mass.value, project=True)
        sig = np.sqrt(lpmodel.v2p)

        dx, dy = calc_xy_offset(ra=self.ra, dec=self.dec, ra_center=ra_center, dec_center=dec_center)
        r = np.sqrt(dx**2 + dy**2)

        f = interpolate.interp1d(lpmodel.r*self.pc2arcmin, sig, fill_value=(sig.max(), sig.min()), bounds_error=False)

        return f(r)*u.km/u.s
    
    def rotation_model(self, v_sys, **kwargs):
        """
        The method calculates the line-of-sight velocity at the positions
        (r, theta) of the available data points.

        As Limepy models are non-rotating, the line-of-sight velocity equals
        the systemic cluster velocity irrespective of position.

        Parameters
        ----------
        v_sys : float
            The constant systemic velocity of the model.
        kwargs
            This model does not use any additional keyword arguments.

        Returns
        -------
        v_los : ndarray
            The model velocity along the line-of-sight at each of the
            individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.rotation_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        return v_sys * np.ones(self.n_data, dtype=np.float64)

    def lnlike(self, values):
        """
        Calculate the log likelihood of the current model given the data.

        It is assumed that the distribution follows a Gaussian distribution.
        Therefore, the probability p of a single measurement (v, v_err) is
        estimated as:

        p = exp{-(v - v0)**2/[2*(v_disp^2 + v_err^2)]}/[2.*(v_disp^2 + v_err^2)]

        Then the log likelihood is then determined by summing over the
        probabilities of all measurements and taking the ln: loglike = ln(sum(p))

        Parameters
        ----------
        values : array_like
            The current values of the model parameters.

        Returns
        -------
        loglike : float
            The log likelihood of the data given the current model.
        """
        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}
        for parameter, value in self.fetch_parameter_values(values).items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            if parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                continue
                # logger.warning('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate functions at positions of measurements
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihood
        return self._calculate_lnlike(v_los=v_los, sigma_los=sigma_los)

    def create_profiles(self, chains, n_burn, n_samples=100, radii=None, filename=None):
        """
        Converts the parameter distributions returned by the MCMC analysis
        into a radial profile of the velocity dispersion.

        Parameters
        ----------
        chains : ndarray
            The chains produced by the MCMC sampler. They should be provided
            as a 3D array, containing the parameters as first index, the steps
            as second index, and the chains as third index.
        n_burn : int
            The number of steps that are ignored at the beginning of each MCMC
            chain.
        n_samples : int
           Number of parameter sets drawn from chain.
        radii : array_like, optional
            The radii at which the profiles should be calculated.
        filename : str, optional
            Name of a csv-file in which the resulting profiles will be stored.

        Returns
        -------
        profile : instance of astropy.table.Table
            The output table will contain the values of the rotation amplitude
            and the dispersion predicted by the model for the requested radii.
            It will further contain the lower and upper 1sigma and 3sigma
            limits for those values.
        """
        # collect parameters
        fitted_models = {}

        '''
        do_later = []
        params = self.parameters.copy()
        for name, parameter in params.items():
            if parameter.fixed:
                if parameter.expr is not None:
                    do_later.append(parameter)
            else:
                parameter.value = u.Quantity(chains[:, n_burn:, i].flatten(), parameter.unit)
                i += 1

        for parameter in do_later:
            parameter.__getstate__()

        for name, parameter in params.items():
            fitted_models[name] = u.Quantity(parameter.value, parameter.unit)
        '''
        parameters = self.sample_chain(chain=chains, n_burn=n_burn, n_samples=n_samples)

        if radii is None:
            radii = np.logspace(-1, 2.5, 50)*u.arcmin
        else:
            radii = u.Quantity(radii)
            if radii.unit is u.dimensionless_unscaled:
                radii = radii*u.arcmin

        sigma = np.zeros((len(radii), n_samples), dtype=np.float64)
        for i, par in enumerate(parameters):

            lpmodel = limepy(phi0=par['phi0'].value, g=par['g'].value, rh=par['rh'].value, M=par['mass'].value, project=True)
            sig = np.sqrt(lpmodel.v2p)

            f = interpolate.interp1d(lpmodel.r*self.pc2arcmin, sig, fill_value=(sig.max(), sig.min()), bounds_error=False)
            sigma[:, i] = f(radii)

        psigma = np.percentile(sigma, [50, 16, 84, 0.15, 99.85], axis=-1)

        profile = Table([
            Table.Column(radii, name='r'),
            Table.Column(psigma[0], name='sigma', unit=u.km / u.s),
            Table.Column(psigma[1], name='sigma_lower_1s', unit=u.km / u.s),
            Table.Column(psigma[2], name='sigma_upper_1s', unit=u.km / u.s),
            Table.Column(psigma[3], name='sigma_lower_3s', unit=u.km / u.s),
            Table.Column(psigma[4], name='sigma_upper_3s', unit=u.km / u.s),
        ])

        if filename is not None:
            profile.write(filename, format='ascii.ecsv', overwrite=True)

        return profile
    

class LimepyFitGB(LimepyFit):

    """
    A child class of LimepyFit that includes a background component
    approximated by a Gaussian in radial velocity space.

    Compared to the parent LimepyFit class, this class uses three additional
    parameters. `v_back` and `sigma_back` are the shape parameters of the
    Gaussian used to describe the distribution of background velocities.
    `f_back` measures the fractional contribution of background sources to
    the observed source density.

    As an additional observable, `density` must be provided, indicating the
    2dim. source density of the target distribution at the position of each
    velocity measurement, relative to a central value.
    """
    MODEL_PARAMETERS = LimepyFit.MODEL_PARAMETERS + ['v_back', 'sigma_back', 'f_back']
    OBSERVABLES = dict(LimepyFit.OBSERVABLES, **{'density': u.dimensionless_unscaled})

    parameters_file = files(config).joinpath('limepy_with_background.json')

    def __init__(self, data, parameters=None, **kwargs):
        """
        Initialize a new instance of the LimepyFitGB class.

        Parameters
        ----------
        data : instance of DataReader
            The observed data for a set of n stars. In addition to the
            observables required to initialize an instance of the parent
            LimepyFit class, the data also need to include a column named
            'density', containing the normalized stellar surface density at
            the location of each star.
        parameters : instance of Parameters, optional
            The model parameters
        kwargs
            Any additional keyword arguments are passed to the initialization
            of the parent class.
        """
        # additionally required observables
        self.density = None

        # No additional background component is currently supported
        background = kwargs.pop('background', None)
        if background is not None:
            logger.error('Class LimepyFitGB does not support additional background components.')

        if parameters is None:
            parameters = Parameters().load(self.parameters_file)

        # call parent class initialisation.
        super(LimepyFitGB, self).__init__(data=data, parameters=parameters, **kwargs)

    def lnlike(self, values):
        """
        Calculate the log likelihood of the current model given the data.

        It is assumed that the distribution follows a Gaussian distribution.
        Therefore, the probability p of a single measurement (v, v_err) is
        estimated as:

        p = exp{-(v - v0)**2/[2*(v_disp^2 + v_err^2)]}/[2.*(v_disp^2 + v_err^2)]

        Then the log likelihood is then determined by summing over the
        probabilities of all measurements and taking the ln: loglike = ln(sum(p))

        Parameters
        ----------
        values : array_like
            The current values of the model parameters.

        Returns
        -------
        loglike : float
            The log likelihood of the data given the current model.
        """
        parameter_dict = self.fetch_parameter_values(values)

        # calculate log-likelihoods for background population
        v_back = parameter_dict.pop('v_back')
        sigma_back = parameter_dict.pop('sigma_back')
        f_back = parameter_dict.pop('f_back')

        norm = self.verr*self.verr + sigma_back*sigma_back
        exponent = -0.5 * np.power(self.v - v_back, 2) / norm

        lnlike_back = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        # get membership priors
        m = self.density/(self.density + f_back)

        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}

        for parameter, value in parameter_dict.items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            if parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                continue
                # logger.warning('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihoods for cluster population
        norm = self.verr * self.verr + sigma_los * sigma_los
        exponent = -0.5 * np.power(self.v - v_los, 2) / norm

        lnlike_cluster = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        max_lnlike = np.max([lnlike_cluster, lnlike_back], axis=0)

        lnlike = max_lnlike + np.log(m*np.exp(lnlike_cluster - max_lnlike) + (1. - m)*np.exp(lnlike_back - max_lnlike))

        return lnlike.sum()

    def calculate_membership_probabilities(self, chain, n_burn):

        bestfit = self.compute_bestfit_values(chain=chain, n_burn=n_burn)
        parameters = dict(zip(bestfit.columns, [bestfit.loc['median'][c] for c in bestfit.columns]))
        _ = parameters.pop('value')

        # add constant parameters
        for name, parameter in self.parameters.items():
            if parameter.fixed:
                parameters[name] = u.Quantity(parameter.value, parameter.unit)

        # calculate log-likelihoods for background population
        v_back = parameters.pop('v_back')
        sigma_back = parameters.pop('sigma_back')
        f_back = parameters.pop('f_back')

        norm = self.verr*self.verr + sigma_back*sigma_back
        exponent = -0.5 * np.power(self.v - v_back, 2) / norm

        lnlike_back = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        # get membership priors
        m = self.density/(self.density + f_back)

        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}

        used = False
        for parameter, value in parameters.items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
                used = True
            if parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
                used = True
            if not used:
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihoods for cluster population
        norm = self.verr * self.verr + sigma_los * sigma_los
        exponent = -0.5 * np.power(self.v - v_los, 2) / norm

        lnlike_cluster = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        max_lnlike = np.max([lnlike_cluster, lnlike_back], axis=0)

        return m*np.exp(lnlike_cluster - max_lnlike) / (
                m*np.exp(lnlike_cluster - max_lnlike) + (1. - m)*np.exp(lnlike_back - max_lnlike))
