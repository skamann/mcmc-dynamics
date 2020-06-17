import inspect
import logging
import numpy as np
from astropy import units as u
from astropy.table import QTable
from .runner import Runner


logger = logging.getLogger(__name__)


class ConstantFit(Runner):

    def __init__(self, data, initials, **kwargs):
        """
        Initialize a new instance of the ConstantFit class.

        Parameters
        ----------
        data : instance of DataReader
            The observed data for a set of n stars. The instance must provide
            at least the velocities and their uncertainties.
        initials : list of dictionaries
            The status of the model parameters in the analysis. For each
             parameter in the analysis, at least the entries 'name',
            'init', and 'fixed' must be provided.
        """
        # required observables
        self.theta = None

        super(ConstantFit, self).__init__(data=data, initials=initials, **kwargs)

        # # get required columns - if units were provided, make sure they are as we expect
        # self.theta = u.Quantity(data.data['theta'])
        # if self.theta.unit.is_unity():
        #     self.theta *= u.rad
        #     logging.warning('Missing unit for <theta> values, assuming {0}.'.format(self.theta.unit))

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters
        self.dispersion_parameters = inspect.signature(self.dispersion_model).parameters

    @property
    def observables(self):
        if self._observables is None:
            self._observables = super(ConstantFit, self).observables
            self._observables['theta'] = u.rad
        return self._observables

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(ConstantFit, self).parameters
            self._parameters.update(
                {'v_sys': u.km / u.s, 'sigma_max': u.km / u.s, 'v_maxx': u.km / u.s, 'v_maxy': u.km/u.s})
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
            # elif row['name'] == 'theta_0':
            #     labels[row['name']] = r'$\theta_{{\rm 0}}/${0}'.format(latex_string)
            elif row['name'] == 'sigma_max':
                labels[row['name']] = r'$\sigma_{{\rm 0}}/${0}'.format(latex_string)
            else:
                labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
        return labels

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
        # Split parameters
        for parameter, value in self.fetch_parameters(values).items():
            if parameter == 'sigma_max' and (value <= 0 or value > 100*u.km/u.s):
                print('{} causes lnprior = -inf'.format(parameter))
                return -np.inf
            elif parameter in ['v_maxx', 'vmaxy'] and abs(value) > 50*u.km/u.s:
                print('{} causes lnprior = -inf'.format(parameter))
                return -np.inf
            # elif parameter == 'theta_0' and (value < 0 or value > np.pi*u.rad):
            #     return -np.inf
        return 0

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

        for parameter, value in self.fetch_parameters(values).items():
            if parameter in self.rotation_parameters.keys():
                kwargs_rotation[parameter] = value
            elif parameter in self.dispersion_parameters.keys():
                kwargs_dispersion[parameter] = value
            else:
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate likelihood
        return self._calculate_lnlike(v_los=v_los, sigma_los=sigma_los)

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
        initials = np.zeros((n_walkers, self.n_fitted_parameters))
        i = 0
        for row in self.initials:
            if row['fixed']:
                continue
            # if row['name'] == 'theta_0':
            #     initials[:, i] = np.pi*u.rad * np.random.rand(n_walkers)
            else:
                initials[:, i] = row['init'] + np.random.randn(n_walkers)*row['init'].unit
            i += 1
        return initials

    def compute_theta_vmax(self, chain, n_burn, return_samples=False):

        try:
            i = self.fitted_parameters.index('v_maxx')
            j = self.fitted_parameters.index('v_maxy')
        except ValueError:
            logger.error("'v_maxx' and/or 'v_maxy' missing in list of fitted parameters.")
            return None

        v_maxx = chain[:, n_burn:, i].flatten()
        v_maxy = chain[:, n_burn:, j].flatten()

        # v_max = np.sqrt(v_maxx ** 2 + v_maxy ** 2)
        theta = np.arctan2(v_maxy, v_maxx)

        median_theta = np.arctan2(np.median(v_maxy), np.median(v_maxx))

        # make sure median angle is in the centre of the full angle range (i.e. at 0 when range is (-Pi, Pi])
        _theta = theta - median_theta
        _theta = np.where(_theta < -np.pi, _theta + 2 * np.pi, _theta)
        _theta = np.where(_theta > np.pi, _theta - 2 * np.pi, _theta)

        # v_max[abs(_theta) > np.pi/2.] *= -1
        v_max = v_maxx * np.cos(-median_theta) - v_maxy * np.sin(-median_theta)

        results = QTable(data=[['median', 'uperr', 'loerr']], names=['value'])
        results.add_index('value')

        for name, values in {'v_max': v_max, 'theta_0': _theta}.items():

            unit = u.rad if name == 'theta_0' else self.units['v_maxx']

            percentiles = np.percentile(values, [16, 50, 84])

            results.add_column(QTable.Column(
                [percentiles[1], percentiles[2] - percentiles[1], percentiles[1] - percentiles[0]]*unit,
                name=name))

        results.loc['median']['theta_0'] += median_theta*u.rad
        
        if return_samples:
            return results, v_max, _theta

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
            logger.error('Class ConstantFitGB does not support additional background components.')

        # call parent class initialisation.
        super(ConstantFitGB, self).__init__(data=data, initials=initials, **kwargs)

    @property
    def observables(self):
        if self._observables is None:
            self._observables = super(ConstantFitGB, self).observables
            self._observables['density'] = u.dimensionless_unscaled
        return self._observables

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(ConstantFitGB, self).parameters
            self._parameters.update(
                {'v_back': u.km / u.s, 'sigma_back': u.km / u.s, 'f_back': u.dimensionless_unscaled})
        return self._parameters

    @property
    def parameter_labels(self):

        labels = super(ConstantFitGB, self).parameter_labels
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
        for parameter, value in self.fetch_parameters(values).items():
            if parameter == 'f_back' and (value < 0 or value > 1):
                return -np.inf
            elif parameter == 'sigma_back' and (value <= 0 or value > 100 * u.km / u.s):
                return -np.inf
        return super(ConstantFitGB, self).lnprior(values)

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
        parameter_dict = self.fetch_parameters(values)

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
                raise IOError('Unknown model parameter "{0}" provided.'.format(parameter))

        # evaluate models of positions of data points
        v_los = self.rotation_model(**kwargs_rotation)
        sigma_los = self.dispersion_model(**kwargs_dispersion)

        # calculate log-likelihoods for cluster population
        norm = self.verr * self.verr + sigma_los * sigma_los
        exponent = -0.5 * np.power(self.v - v_los, 2) / norm

        lnlike_cluster = -0.5 * np.log(2. * np.pi * norm.value) + exponent

        return lnlike_cluster, lnlike_back, m

    def get_initials(self, n_walkers):

        initials = super(ConstantFitGB, self).get_initials(n_walkers)

        i = 0
        for row in self.initials:
            if row['fixed']:
                continue
            if row['name'] == 'f_back':
                initials[:, i] = np.random.random_sample(n_walkers)
            i += 1

        return initials

    def calculate_membership_probabilities(self, chain, n_burn):

        bestfit = self.compute_bestfit_values(chain=chain, n_burn=n_burn)
        parameters = dict(zip(bestfit.columns, [bestfit.loc['median'][c] for c in bestfit.columns]))
        _ = parameters.pop('value')

        lnlike_cluster, lnlike_back, m = self._calculate_lnlike_cluster_back(parameters)

        return m*np.exp(lnlike_cluster) / (m*np.exp(lnlike_cluster) + (1. - m)*np.exp(lnlike_back))
