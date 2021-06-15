# -*- coding: utf-8 -*-
import inspect
import logging
import numpy as np
import importlib.resources as pkg_resources
from astropy import units as u
from astropy.table import Table
from .runner import Runner
from .. import config
from ..parameter import Parameters

logger = logging.getLogger(__name__)


class ModelFit(Runner):
    """
    The purpose of the ModelFit class is to fit the radial rotation and
    dispersion profile of a cluster using simple analytical models.

    The rotation profile is modeled as expected for a system that underwent
    violent relaxation (e.g. Lynden-Bell 1967, MNRAS, 136, 101). In this case,
    the radial dependence is given as

    v_rot(r, theta) = V_SYS + 2*(V_MAX/R_PEAK) * x_pa/(1. + (x_pa/R_PEAK)**2),

    where

    x_pa(r, theta) = r * np.sin(theta - THETA_0).


    The dispersion is modeled as a Plummer (1911, MNRAS, 71, 460) profile with
    the following functional form,

    sigma(r) = SIGMA_0/(1. + r**2 / A**2)**0.25.


    Hence, the model has up to 6 free parameters, V_SYS, V_MAX, R_PEAK,
    THETA_0, SIGMA_0, and A.

    The data required per star are the distance r to the cluster centre, the
    position angle theta (measured from north counterclockwise), the radial
    velocity v and the velocity uncertainty epsilon_v.
    """
    MODEL_PARAMETERS = ['v_sys', 'v_maxx', 'v_maxy', 'r_peak', 'sigma_max', 'a']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'r': u.arcsec, 'theta': u.rad}

    def __init__(self, data, parameters=None, **kwargs):
        """
        Initialize a new instance of the ModelFit class.

        Parameters
        ----------
        data : instance of DataReader
            The observed data for a set of n stars. The instance must provide
            at least the radii, the position angles, the velocities, and their
            uncertainties.
        parameters: instance of Parameters, optional
            The model parameters.
        kwargs :
            Any additional keyword arguments are passed on to the
            initialization of the parent class.
        """
        # required observables
        self.r = None
        self.theta = None

        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'model.json'))

        super(ModelFit, self).__init__(data=data, parameters=parameters, **kwargs)

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters
        self.dispersion_parameters = inspect.signature(self.dispersion_model).parameters

    def dispersion_model(self, sigma_max, a=1, **kwargs):
        """
        The method calculates the line-of-sight velocity dispersion at the
        positions (r, theta) of the available data points.

        In this model, the line-of-sight velocity dispersion is calculated
        as follows.

        sigma_los = sigma_max / (1. + r^2 / a^2)^0.25

        Parameters
        ----------
        sigma_max : float
            The central velocity dispersion of the model.
        a : float
            The scale radius of the model.
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

        return sigma_max / (1. + self.r ** 2 / a ** 2) ** 0.25

    def rotation_model(self, v_sys, v_maxx, v_maxy, r_peak=1., **kwargs):
        """
        The method calculates the line-of-sight velocity at the positions
        (r, theta) of the available data points.

        In this model, the line-of-sight velocity is calculated from the
        systemic velocity and the rotation model as follows.

        v_los = v_sys + 2*(v_max/r_peak)*x_pa/(1. + (x_pa / r_peak)^2),

        with x_pa = r*sin(theta - theta_0).
             v_max = sqrt(v_maxx^2 + v_maxy^2)
             theta_0 = arctan(v_maxy/v_maxx)

        Parameters
        ----------
        v_sys : float
            The constant systemic velocity of the model.
        v_maxx : float
            The x-component of the rotation amplitude of the model.
        v_maxy : float
            The y-component of the rotation amplitude of the model.
        r_peak : float
            The position of the peak of the rotation curve.
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

        v_max = np.sqrt(v_maxx**2 + v_maxy**2)
        theta_0 = np.arctan2(v_maxy, v_maxx)

        x_pa = self.r * np.sin(self.theta - theta_0)
        return v_sys + 2. * (v_max / r_peak) * x_pa / (1. + (x_pa / r_peak) ** 2)

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
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate functions at positions of measurements
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihood
        return self._calculate_lnlike(v_los=v_los, sigma_los=sigma_los)

    def create_profiles(self, chains, n_burn, radii=None, filename=None):
        """
        Converts the parameter distributions returned by the MCMC analysis
        into radial profiles of the rotation amplitude and velocity
        dispersion.

        Parameters
        ----------
        chains : ndarray
            The chains produced by the MCMC sampler. They should be provided
            as a 3D array, containing the parameters as first index, the steps
            as second index, and the chains as third index.
        n_burn : int
            The number of steps that are ignored at the beginning of each MCMC
            chain.
        radii : array_like, optional
            The radii at which the profiles should be calculated.
        filename : str, optional
            Name of a csv-file in which the resulting profiles will be stored.

        Returns
        -------

        """
        # collect parameters
        fitted_models = {}

        i = 0
        for name, parameter in self.parameters.items():
            if parameter.fixed:
                fitted_models[name] = u.Quantity(parameter.value, parameter.unit)
            else:
                fitted_models[name] = u.Quantity(chains[:, n_burn:, i].flatten(), parameter.unit)
                i += 1

        v_maxx = fitted_models['v_maxx']
        v_maxy = fitted_models['v_maxy']
        r_peak = fitted_models['r_peak']
        sigma_max = fitted_models['sigma_max']
        a = fitted_models['a']

        if radii is None:
            radii = np.logspace(-1, 2.5, 50)*u.arcsec
        else:
            radii = u.Quantity(radii)
            if radii.unit is u.dimensionless_unscaled:
                radii = radii*r_peak.unit

        v_max = np.sqrt(v_maxx**2 + v_maxy**2)
        v_rot = 2. * (v_max / r_peak) * radii[:, np.newaxis] / (1. + (radii[:, np.newaxis] / r_peak) ** 2)
        pv_rot = np.percentile(v_rot.to(u.km/u.s), [50, 16, 84, 0.15, 99.85], axis=-1)
        sigma = sigma_max / (1. + radii[:, np.newaxis] ** 2 / a ** 2) ** 0.25
        psigma = np.percentile(sigma.to(u.km/u.s), [50, 16, 84, 0.15, 99.85], axis=-1)

        profile = Table([
            Table.Column(radii, name='r'),
            Table.Column(pv_rot[0], name='v_rot', unit=u.km/u.s),
            Table.Column(pv_rot[1], name='v_rot_lower_1s', unit=u.km / u.s),
            Table.Column(pv_rot[2], name='v_rot_upper_1s', unit=u.km / u.s),
            Table.Column(pv_rot[3], name='v_rot_lower_3s', unit=u.km / u.s),
            Table.Column(pv_rot[4], name='v_rot_upper_3s', unit=u.km / u.s),
            Table.Column(psigma[0], name='sigma', unit=u.km / u.s),
            Table.Column(psigma[1], name='sigma_lower_1s', unit=u.km / u.s),
            Table.Column(psigma[2], name='sigma_upper_1s', unit=u.km / u.s),
            Table.Column(psigma[3], name='sigma_lower_3s', unit=u.km / u.s),
            Table.Column(psigma[4], name='sigma_upper_3s', unit=u.km / u.s),
        ])

        if filename is not None:
            profile.write(filename, format='ascii.ecsv', overwrite=True)

        return profile


class ModelFitGB(ModelFit):

    """
    A child class of ModelFit that includes a background component
    approximated by a Gaussian in radial velocity space.

    Compared to the parent ModelFit class, this class uses three additional
    parameters. `v_back` and `sigma_back` are the shape parameters of the
    Gaussian used to describe the distribution of background velocities.
    `f_back` measures the fractional contribution of background sources to
    the observed source density.

    As an additional observable, `density` must be provided, indicating the
    2dim. source density of the target distribution at the position of each
    velocity measurement, relative to a central value.
    """
    MODEL_PARAMETERS = ['v_back', 'sigma_back', 'f_back', 'v_sys', 'v_maxx', 'v_maxy', 'r_peak', 'sigma_max', 'a']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'r': u.arcsec, 'theta': u.rad, 'density': u.dimensionless_unscaled}

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
            logger.error('Class ConstantFitGB does not support additional background components.')

        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'model_with_background.json'))

        # call parent class initialisation.
        super(ModelFitGB, self).__init__(data=data, parameters=parameters, **kwargs)

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
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

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

        for parameter, value in parameters.items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
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


class ModelFitNonGB(ModelFit):

    """
    A child class of ModelFit that includes a background component
    approximated by a Gaussian in radial velocity space.

    Compared to the parent ModelFit class, this class uses three additional
    parameters. `v_back` and `sigma_back` are the shape parameters of the
    Gaussian used to describe the distribution of background velocities.
    `f_back` measures the fractional contribution of background sources to
    the observed source density.

    As an additional observable, `density` must be provided, indicating the
    2dim. source density of the target distribution at the position of each
    velocity measurement, relative to a central value.
    """
    MODEL_PARAMETERS = ['f_back', 'v_sys', 'v_maxx', 'v_maxy', 'r_peak', 'sigma_max', 'a']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'r': u.arcsec, 'theta': u.rad, 'density': u.dimensionless_unscaled}

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
        parameters : instance of Parameters, optional
            The model parameters
        kwargs
            Any additional keyword arguments are passed to the initialization
            of the parent class.
        """
        # additionally required observables
        self.density = None

        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'model_with_background.json'))

        background = kwargs.pop('background', None)

        # call parent class initialisation.
        super(ModelFitNonGB, self).__init__(data=data, parameters=parameters, **kwargs)

        self.background = background
        self.lnlike_background = self.background(self.v, self.verr)

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
        f_back = parameter_dict.pop('f_back')

        # get membership priors
        m = self.density/(self.density + f_back)

        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}

        for parameter, value in parameter_dict.items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihoods for cluster population
        norm = self.verr * self.verr + sigma_los * sigma_los
        exponent = -0.5 * np.power(self.v - v_los, 2) / norm

        lnlike_cluster = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        max_lnlike = np.max([lnlike_cluster, self.lnlike_background], axis=0)

        lnlike = max_lnlike + np.log(m*np.exp(lnlike_cluster - max_lnlike)
                                     + (1. - m)*np.exp(self.lnlike_background - max_lnlike))
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
        f_back = parameters.pop('f_back')

        # get membership priors
        m = self.density/(self.density + f_back)

        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}

        for parameter, value in parameters.items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihoods for cluster population
        norm = self.verr * self.verr + sigma_los * sigma_los
        exponent = -0.5 * np.power(self.v - v_los, 2) / norm

        lnlike_cluster = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        max_lnlike = np.max([lnlike_cluster, self.lnlike_background], axis=0)

        return m*np.exp(lnlike_cluster - max_lnlike) / (
                m*np.exp(lnlike_cluster - max_lnlike) + (1. - m)*np.exp(self.lnlike_background - max_lnlike))
