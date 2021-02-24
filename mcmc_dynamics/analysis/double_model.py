# -*- coding: utf-8 -*-
import inspect
import logging
import numpy as np
from astropy import units as u
from astropy.table import Table
from .model import ModelFit


logger = logging.getLogger(__name__)


class DoubleModelFit(ModelFit):
    """
    The DoubleModelFit class is an extension to the ModelFit class, which uses
    the same analytical prescription for fitting the radial dispersion profile
    (i.e. a Plummer profile), but instead of using a single Lynden-Bell model,
    the rotation profile is fitted using a double Lynden-Bell profile.

    The radial dependence of the rotation profile in this case is as follows.

    v_rot(r, theta) = V_SYS + \
        2*(V_MAX/R_PEAK) * x_pa/(1. + (x_pa/R_PEAK)**2) + \
        2*(V_MAX_c/R_PEAK_c) * x_pa_c/(1. + (x_pa_c/R_PEAK_c)**2),

    where

    x_pa(r, theta) = r * np.sin(theta - THETA_0) and
    x_pa_c(r, theta) = r * np.sin(theta - THETA_0_c).

    Together with the free parameters defining the dispersion profile
    (SIGMA_0, A), the model has up to 9 free parameters, V_SYS, V_MAX, R_PEAK,
    THETA_0, V_MAX_c, R_PEAK_c, THETA_0_c, SIGMA_0, and A.

    The data required per star are the distance r to the cluster centre, the
    position angle theta (measured from north counterclockwise), the radial
    velocity v and the velocity uncertainty epsilon_v.
    """

    def __init__(self, data, initials, **kwargs):
        """
        Initialize a new instance of the ModelFit class.

        Parameters
        ----------
        data : instance of DataReader
            The observed data for a set of n stars. The instance must provide
            at least the radii, the position angles, the velocities, and their
            uncertainties.
        initials : list of dictionaries, optional
            For each model parameter a separate dictionary must be available,
            containing the keys 'name', 'init', and 'fixed'.
        kwargs :
            Any additional keyword arguments are passed on to the
            initialization of the parent class.
        """
        super(DoubleModelFit, self).__init__(data=data, initials=initials, **kwargs)

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters

    @property
    def observables(self):
        if self._observables is None:
            self._observables = super(DoubleModelFit, self).observables
            self._observables.update({'r': u.arcsec, 'theta': u.rad})
        return self._observables

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(DoubleModelFit, self).parameters
            self._parameters.update({'v_maxx_c': u.km / u.s, 'v_maxy_c': u.km / u.s, 'r_peak_c': u.arcmin})
        return self._parameters

    @property
    def parameter_labels(self):
        labels = {}
        for row in self.initials:
            latex_string = row['init'].unit.to_string('latex')
            if row['name'] == 'v_sys':
                labels[row['name']] = r'$v_{{\rm sys}}/${0}'.format(latex_string)
            elif row['name'] == 'v_maxx':
                labels[row['name']] = r'$v_{{\rm max,\,x}}/${0}'.format(latex_string)
            elif row['name'] == 'v_maxy':
                labels[row['name']] = r'$v_{{\rm max,\,y}}/${0}'.format(latex_string)
            elif row['name'] == 'r_peak':
                labels[row['name']] = r'$r_{{\rm peak}}/${0}'.format(latex_string)
            # elif row['name'] == 'theta_0':
            #     labels[row['name']] = r'$\theta_{{\rm 0}}/${0}'.format(latex_string)
            elif row['name'] == 'sigma_max':
                labels[row['name']] = r'$\sigma_{{\rm 0}}/${0}'.format(latex_string)
            elif row['name'] == 'a':
                labels[row['name']] = r'$a/${0}'.format(latex_string)
            else:
                labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
        return labels

    def rotation_model(self, v_sys, v_maxx, v_maxy, r_peak=1., v_maxx_c=0., v_maxy_c=0., r_peak_c=0., **kwargs):
        """
        The method calculates the line-of-sight velocity at the positions
        (r, theta) of the available data points.

        In this model, the line-of-sight velocity is calculated from the
        systemic velocity and the rotation model as follows.

        v_los = v_sys + 2*(v_max/r_peak)*x_pa/(1. + (x_pa / r_peak)^2) + \
            2*(v_max_c/r_peak_c)*x_pa_c/(1. + (x_pa_c / r_peak_c)^2),

        with x_pa = r*sin(theta - theta_0),
             v_max = sqrt(v_maxx^2 + v_maxy^2),
             theta_0 = arctan(v_maxy/v_maxx),
             x_pa_c = r*sin(theta - theta_0_c),
             v_max_c = sqrt(v_maxx_c^2 + v_maxy_c^2),
             theta_0_c = arctan(v_maxy_c/v_maxx_c)

        Parameters
        ----------
        v_sys : float
            The constant systemic velocity of the model.
        v_maxx : float
            The x-component of the rotation amplitude of the primary model
            curve.
        v_maxy : float
            The y-component of the rotation amplitude of the primary model
            curve.
        r_peak : float
            The position of the peak of the primary model curve.
        v_maxx_c : float
            The x-component of the rotation amplitude of the additional model
            curve.
        v_maxy_c : float
            The y-component of the rotation amplitude of the additional model
            curve.
        r_peak_c : float
            The position of the peak of the additional rotation curve.
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

        v_max_c = np.sqrt(v_maxx_c**2 + v_maxy_c**2)
        theta_0_c = np.arctan2(v_maxy_c, v_maxx_c)

        x_pa = self.r * np.sin(self.theta - theta_0)
        x_pa_c = self.r * np.sin(self.theta - theta_0_c)
        return v_sys + 2. * (v_max / r_peak) * x_pa / (1. + (x_pa / r_peak) ** 2) + 2. * (
                v_max_c / r_peak_c) * x_pa_c / (1. + (x_pa_c / r_peak_c) ** 2)

    def lnprior(self, values):
        """
        Check if the priors for the model parameters are fulfilled.

        This method implements the priors needed for the MCMC estimation
        of the uncertainties. Uninformative priors are used, i.e. the
        likelihoods are constant across the accepted value range and zero
        otherwise.

        Parameters
        ----------
        values : array_like
            The current values of the model parameters.

        Returns
        -------
        loglike : float
            The log likelihood of the model for the given parameters. As
            uninformative priors are used, the log likelihood will be zero
            for valid parameters and -inf otherwise.
        """
        parameters = self.fetch_parameter_values(values)

        for parameter, value in parameters.items():
            if parameter in ['v_maxx_c', 'v_maxy_c'] and abs(value) > 50*u.km/u.s:
                return -np.inf
            elif parameter in 'r_peak_c' and not 0 < value <= parameters['r_peak']:
                return -np.inf
        return super(DoubleModelFit, self).lnprior(values)

    def get_initials(self, n_walkers):
        """
        Create initial values for the MCMC chains.

        Parameters
        ----------
        n_walkers : int
            The number of walkers for which initial guesses should be created.

        Returns
        -------
        initials : ndarray
            The initial guesses for the requested number of chains.
        """
        # define initial positions of the walkers in parameter space
        initials = super(DoubleModelFit, self).get_initials(n_walkers)

        i = 0
        for row in self.initials:
            if row['fixed']:
                continue
            elif row['name'] in ['r_peak_c']:
                initials[:, i] = row['init'] * np.random.lognormal(0.0, 0.2, n_walkers)
            i += 1

        return initials

    def create_profiles(self, chains, n_burn, filename=None):
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
        filename : str, optional
            Name of a csv-file in which the resulting profiles will be stored.

        Returns
        -------

        """
        # collect parameters
        fitted_models = {}

        i = 0
        for row in self.initials:
            if row['fixed']:
                fitted_models[row['name']] = row['init']
            else:
                fitted_models[row['name']] = chains[:, n_burn:, i].flatten()*row['init'].unit
                i += 1

        v_maxx = fitted_models['v_maxx']
        v_maxy = fitted_models['v_maxy']
        r_peak = fitted_models['r_peak']

        v_maxx_c = fitted_models['v_maxx_c']
        v_maxy_c = fitted_models['v_maxy_c']
        r_peak_c = fitted_models['r_peak_c']

        sigma_max = fitted_models['sigma_max']
        a = fitted_models['a']

        radii = np.logspace(-1, 2.5, 50)*u.arcsec

        v_max = np.sqrt(v_maxx**2 + v_maxy**2)
        v_max_c = np.sqrt(v_maxx_c**2 + v_maxy_c**2)

        v_rot = 2. * (v_max / r_peak) * radii[:, np.newaxis] / (1. + (radii[:, np.newaxis] / r_peak) ** 2)
        v_rot_c = 2. * (v_max_c / r_peak_c) * radii[:, np.newaxis] / (1. + (radii[:, np.newaxis] / r_peak_c) ** 2)
        pv_rot = np.percentile((v_rot + v_rot_c).to(u.km/u.s), [50, 16, 84, 0.15, 99.85], axis=-1)

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


class DoubleModelFitGB(DoubleModelFit):

    """
    A child class of ModelFit that includes a background component
    approximated by a Gaussian in radial velocity space.
    """
    def __init__(self, data, initials, **kwargs):
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
        initials : list of dictionaries
            The status of the model parameters in the analysis. For each
             parameter in the analysis, at least the entries 'name',
            'init', and 'fixed' must be provided.
        kwargs
            Any additional keyword arguments are passed to the initialization
            of the parent class.
        """
        # additionally required observables
        self.density = None

        # No additional background component is currently supported
        background = kwargs.pop('background', None)
        if background is not None:
            logger.error('Class DoubleModelFitGB does not support additional background components.')

        # call parent class initialisation.
        super(DoubleModelFitGB, self).__init__(data=data, initials=initials, **kwargs)

    @property
    def observables(self):
        if self._observables is None:
            self._observables = super(DoubleModelFitGB, self).observables
            self._observables['density'] = u.dimensionless_unscaled
        return self._observables

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(DoubleModelFitGB, self).parameters
            self._parameters.update(
                {'v_back': u.km / u.s, 'sigma_back': u.km / u.s, 'f_back': u.dimensionless_unscaled})
        return self._parameters

    @property
    def parameter_labels(self):

        labels = super(DoubleModelFitGB, self).parameter_labels
        for row in self.initials:
            latex_string = row['init'].unit.to_string('latex')
            if row['name'] == 'v_back':
                labels[row['name']] = r'$v_{{\rm back}}/${0}'.format(latex_string)
            elif row['name'] == 'sigma_back':
                labels[row['name']] = r'$\sigma_{{\rm back}}/${0}'.format(latex_string)
            elif row['name'] == 'f_back':
                labels[row['name']] = r'$f_{\rm back}$'
        return labels

    def lnprior(self, values):
        for parameter, value in self.fetch_parameter_values(values).items():
            if parameter == 'f_back' and (value < 0 or value > 1):
                return -np.inf
            elif parameter == 'sigma_back' and (value <= 0 or value > 100 * u.km / u.s):
                return -np.inf
        return super(DoubleModelFitGB, self).lnprior(values)

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

    def get_initials(self, n_walkers):

        initials = super(DoubleModelFitGB, self).get_initials(n_walkers)

        i = 0
        for row in self.initials:
            if row['fixed']:
                continue
            if row['name'] == 'f_back':
                initials[:, i] = 2.*row['init']*np.random.random_sample(n_walkers)
            i += 1

        return initials

    def calculate_membership_probabilities(self, chain, n_burn):

        bestfit = self.compute_bestfit_values(chain=chain, n_burn=n_burn)
        parameters = dict(zip(bestfit.columns, [bestfit.loc['median'][c] for c in bestfit.columns]))
        _ = parameters.pop('value')

        # add constant parameters
        for row in self.initials:
            if row['fixed']:
                parameters[row['name']] = row['init']

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
