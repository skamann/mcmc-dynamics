import inspect
import logging
import numpy as np
import importlib.resources as pkg_resources
from astropy import units as u
from astropy.table import QTable
from .runner import Runner
from .. import config
from ..parameter import Parameters
from ..utils.coordinates import get_amplitude_and_angle


logger = logging.getLogger(__name__)


class ConstantFit(Runner):

    MODEL_PARAMETERS = ['v_sys', 'sigma_max', 'v_maxx', 'v_maxy']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'theta': u.rad}

    def __init__(self, data, parameters=None, **kwargs):
        """
        Initialize a new instance of the ConstantFit class.

        Parameters
        ----------
        data : instance of DataReader
            The observed data for a set of n stars. The instance must provide
            at least the velocities and their uncertainties.
        parameters : instance of Parameters, optional
            The model parameters.
        kwargs
            Any extra keyword arguments are forwarded to the initialization of
            the super-class.
        """
        # required observables
        self.theta = None

        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'constant.json'))

        super(ConstantFit, self).__init__(data=data, parameters=parameters, **kwargs)

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters
        self.dispersion_parameters = inspect.signature(self.dispersion_model).parameters

    def dispersion_model(self, sigma_max, **kwargs):
        """
        The method calculates the velocity dispersion at the positions of the
        available data points.

        Parameters
        ----------
        sigma_max : float
            The velocity dispersion is assumed to be constant in this model.
        kwargs
            This method does not use any additional keyword arguments.

        Returns
        -------
        sigma_los : ndarray
            The model values of the velocity dispersion at the positions of
            the individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.dispersion_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        return sigma_max*np.ones(self.n_data, dtype=np.float64)

    def rotation_model(self, v_sys, v_maxx, v_maxy, **kwargs):
        """
        The method calculates the rotation velocity at the positions of the
        available data points.

        Parameters
        ----------
        v_sys : float
            The constant systemic velocity of the model.
        v_maxx : float
            The x-component of the constant rotation velocity of the model.
        v_maxy : float
            The y-component of the constant rotation velocity of the model.
        kwargs
            This model does not use any additional keyword arguments.

        Returns
        -------
        v_los : ndarrray
             The values of the rotation velocity at the positions of the
             individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.rotation_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        v_max = np.sqrt(v_maxx**2 + v_maxy**2)
        theta_0 = np.arctan2(v_maxy, v_maxx)
        
        return v_sys + v_max*np.sin(self.theta - theta_0)

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
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                continue
                # raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate likelihood
        return self._calculate_lnlike(v_los=v_los, sigma_los=sigma_los)

    def compute_theta_vmax(self, chain, n_burn, return_samples=False):
        """
        Compute the position angle `theta_0` and the amplitude of the rotation
        field, `v_max`.

        For each set of parameters available in the provided chain (ignoring
        a using-provided number of steps at the beginning of each walker as
        burn-in), the code will determine the values of `theta_0` and `v_max`
        Afterwards, the median and the 16th and 84th percentiles of the
        distributions thereby obtained are calculated and returned.

        Note that the position angle if measured from north through east and
        gives the orientation of the rotation axis.

        Parameters
        ----------
        chain : ndarray
            The chain returned by the MCMC analysis. Must be a 3dimensional
            array with the different walkers as 0th index, the steps as 1st
            index, and the parameters as 2nd index.
        n_burn : int, optional
            The burn-in for each walker that is discarded when calculating
            the parameter statistics.
        return_samples : bool, optional
            Flag indicating if the full sets of calculated `theta_0` and
            `v_max` values should be returned. By default, only the median
            and 16th and 84th percentiles of each parameter are returned.

        Returns
        -------
        results : instance of astropy.table.QTable
            For each parameter, the table contains one column, providing the
            calculated median, 84th, and 16th percentile in three rows.
        v_max : ndarray
            The calculated amplitude values for all available parameter sets.
            Only returned if `return_samples` is set to True.
        _theta : ndarray
            The calculated position angle values for all available parameter
            sets. Only returned if `return_samples` is set to True.
        sigmas : ndarray
            The available dispersion values. They are not used in any
            calculation and only returned for consistency with other methods.
            Only returned if `return_samples` is set to True.
        """
        pars = self.convert_to_paramaters(chain=chain, n_burn=n_burn)

        results, v_max, _theta = get_amplitude_and_angle(pars, return_samples=return_samples)

        if results is None:
            logger.error('Could not recover paramaters of rotation field in {}.compute_theta_vmax().'.format(
                self.__class__.__name__))
            return None
        else:
            results['v_max'] *= self.units['v_maxx']

        if return_samples:
            return results, v_max, _theta, pars['sigma']
        else:
            return results

    # def leastsq(self, **kwargs):
    #     """
    #     Get best-fit for model parameters using non-linear least squares
    #     fitting.
    #
    #     Parameters
    #     ----------
    #     kwargs
    #         Any keyword arguments input to this method are used as initial
    #         guesses for the parameters to be optimized.
    #
    #     Returns
    #     -------
    #     best_fit : OptimizeResult
    #         The optimization result represented as a OptimizeResult object.
    #         Important attributes are: x the solution array, success a Boolean
    #         flag indicating if the optimizer exited successfully and message
    #         which describes the cause of the termination. See
    #         scipy.optimize.OptimizeResult for a description of other attributes.
    #     """
    #     # check if any initial guesses were provided as keyword arguments
    #     # if not, use default values
    #     initial_guess = np.zeros(len(self.parameters) - len(self.fixed), dtype=np.float32)
    #     for i, prm in enumerate(self.fitted_parameters):
    #         initial_guess[i] = kwargs.pop(prm, self.default_values[prm])
    #     # make sure that no additional keyword arguments were provided
    #     if kwargs:
    #         raise IOError('Unknown keyword argument(s) provided: {0}'.format(kwargs))
    #
    #     # perform least-squares analysis
    #     nll = lambda *args: -1. * self.lnlike(*args)
    #     return minimize(nll, initial_guess, method='Nelder-Mead')


class ConstantFitGB(ConstantFit):
    """
    A child class of ConstantFit that includes a background component
    approximated by a Gaussian in radial velocity space.
    """

    MODEL_PARAMETERS = ['v_back', 'sigma_back', 'f_back', 'v_sys', 'sigma_max', 'v_maxx', 'v_maxy']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'theta': u.rad, 'density': u.dimensionless_unscaled}

    def __init__(self, data, parameters=None, **kwargs):
        """
        Initialize a new instance of the ConstantFitGB class.

        Parameters
        ----------
        data : instance of DataReader
            The observed data for a set of n stars. In addition to the
            observables required to initialize an instance of the parent
            ConstantFit class, the data also need to include a column named
            'density', containing the normalized stellar surface density at
            the location of each star.
        parameters : instance of Parameters
            The model parameters.
        kwargs
            Any additional keyword arguments are passed to the initialization
            of the parent class.
        """
        # additionally required observables
        self.density = None

        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'constant_with_background.json'))

        # No additional background component is currently supported
        background = kwargs.pop('background', None)
        if background is not None:
            logger.error('Class ConstantFitGB does not support additional background components.')

        # call parent class initialisation.
        super(ConstantFitGB, self).__init__(data=data, parameters=parameters, **kwargs)

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

        lnlike_cluster, lnlike_back, m = self._calculate_lnlike_cluster_back(parameter_dict)

        max_lnlike = np.max([lnlike_cluster, lnlike_back], axis=0)

        lnlike = max_lnlike + np.log(m*np.exp(lnlike_cluster - max_lnlike) + (1. - m)*np.exp(lnlike_back - max_lnlike))
        return lnlike.sum()

    def _calculate_lnlike_cluster_back(self, parameters):

        # calculate log-likelihoods for background population
        v_back = parameters.pop('v_back')
        sigma_back = parameters.pop('sigma_back')
        f_back = parameters.pop('f_back')

        norm = self.verr * self.verr + sigma_back * sigma_back
        exponent = -0.5 * np.power(self.v - v_back, 2) / norm

        lnlike_back = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        # get membership priors
        m = self.density / (self.density + f_back)

        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}

        for parameter, value in parameters.items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                continue
                # raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihoods for cluster population
        norm = self.verr * self.verr + sigma_los * sigma_los
        exponent = -0.5 * np.power(self.v - v_los, 2) / norm

        lnlike_cluster = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        return lnlike_cluster, lnlike_back, m

    def calculate_membership_probabilities(self, chain, n_burn):

        bestfit = self.compute_bestfit_values(chain=chain, n_burn=n_burn)
        parameters = dict(zip(bestfit.columns, [bestfit.loc['median'][c] for c in bestfit.columns]))
        _ = parameters.pop('value')

        lnlike_cluster, lnlike_back, m = self._calculate_lnlike_cluster_back(parameters)

        return m*np.exp(lnlike_cluster) / (m*np.exp(lnlike_cluster) + (1. - m)*np.exp(lnlike_back))
