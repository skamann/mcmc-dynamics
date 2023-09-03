import inspect
import logging
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np
import corner
import emcee
from pathos.multiprocessing import Pool
from matplotlib import gridspec
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from astropy import units as u
from astropy.table import QTable

from ..parameter import Parameters
from ..utils.files.data_reader import DataReader


logger = logging.getLogger(__name__)


class Runner(object):
    """
    This class serves as parent class for any of the classes provided to
    analyse the internal kinematics of a stellar system.

    Classes that inherit from `Runner` must at least specify the observables
    and model parameters required by the analysis (see variables `OBSERVABLES`
    and `MODEL_PARAMETERS`). Further, they need to define their own method
    {py:meth}`calculate_model_moments()<analysis.runner.Runner.calculate_model_moments>`,
    which takes a dictionary of parameter values as input in order to
    calculate and return the first and second order moments predicted by the
    model at the locations of the observations.
    """

    MODEL_PARAMETERS = []
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s}

    parameters_file = None

    def __init__(self, data: DataReader, parameters: Parameters, seed: int = 123, background: callable = None,
                 lnlike_background: np.ndarray = None, **kwargs):
        """
        :param data: The observed data.
        :param parameters: The model parameters.
        :param seed: The seed used to initialize the random number generator.
        :param background: The callable used to calculate the likelihood of
            the background population.
        :param lnlike_background: In case a fixed background population is
            used, the log-likelihoods of the stars belonging to the background
            can be provided directly.
        :param kwargs: This method does not take any additional keyword
            arguments.
        """
        # check if any unsupported keyword arguments were provided
        assert not kwargs, "Unknown keyword arguments provided: {0}".format(kwargs)

        # Reproducible results!
        np.random.seed(seed)

        # required observables
        self.v = None
        self.verr = None
        self.density = None  # Only for models including background component

        # sanity checks on data
        assert isinstance(data, DataReader), "'data' must be instance of {0}".format(DataReader.__module__)
        self.data = data

        # can only use background OR lnlike_background
        if background is not None and lnlike_background is not None:
            raise NotImplementedError('Cannot use a constant and a variable background at the same time.')

        # in case a background is included in the model, we need a membership prior and new model parameters
        self.background = background
        if background is not None:
            self.OBSERVABLES['density'] = u.dimensionless_unscaled

            # check calling sequence of background function. Any input parameter that is not an observable is
            # considered as additional model parameter.
            sig = inspect.signature(self.background)
            self.background_parameters = []
            for parameter in sig.parameters.keys():
                if parameter in self.OBSERVABLES.keys() or parameter == 'kwargs':
                    continue
                self.background_parameters.append(parameter)
            self.MODEL_PARAMETERS.extend(self.background_parameters)
            logger.info(f'Background function provided requires the following parameters: {self.background_parameters}')
        else:
            self.background_parameters = None

        if lnlike_background is not None:
            if not len(lnlike_background) == self.n_data:
                raise IOError('Number of background likelihoods does not match number of observations.')
            self.lnlike_background = lnlike_background
        else:
            self.lnlike_background = None

        # make sure all required columns are available
        for required, unit in self.OBSERVABLES.items():
            if required not in data.data.columns:
                raise IOError("Input data missing required column <{0}>".format(required))
            quantity = u.Quantity(data.data[required])
            if quantity.unit.is_unity() and not unit.is_unity():
                quantity *= unit
                logger.warning('Missing units for <{0}> values. Assuming {1}.'.format(required, unit))
            setattr(self, required, quantity)

        # sanity checks on parameters
        if not isinstance(parameters, Parameters):
            raise IOError("'parameters' must be instance of {0}".format(Parameters.__module__))
        self.parameters = parameters

        missing = set(self.MODEL_PARAMETERS).difference(self.parameters)
        if missing:
            raise IOError("Missing required parameter(s): '{0}'".format(missing))

        unused = set(self.parameters).difference(self.MODEL_PARAMETERS)
        if unused:
            logger.warning("Superfluous parameter(s) provided: '{0}'".format(unused))

    @classmethod
    def default_parameters(cls) -> Parameters:
        """
        :return: The default parameters for this class.
        """
        if cls.parameters_file is None:
            raise NotImplementedError

        return Parameters().load(open(cls.parameters_file))

    @property
    def n_data(self) -> int:
        """
        :return: The number of data points in the instance.
        """
        return self.data.sample_size

    @property
    def fitted_parameters(self) -> list:
        """
        :return: The names of the fitted parameters.
        """
        return [p for p in self.parameters if not self.parameters[p].fixed]

    @property
    def n_fitted_parameters(self):
        """
        :return: The number of fitted parameters
        """
        return len(self.fitted_parameters)

    @property
    def units(self) -> dict:
        """
        :return: The units of the fitted parameters
        """
        return {p: self.parameters[p].unit for p in self.parameters}

    def fetch_parameter_values(self, values: np.ndarray) -> dict:
        """
        Collect all current model parameters (fixed or considered for
        optimization) and store them in a dictionary.

        :param values: The current values of the model parameters.

        :return: A dictionary containing one value per model parameter,
            regardless of whether the parameter is fixed or not.
        """
        current_parameters = {}

        i = 0
        for name, parameter in self.parameters.items():
            if parameter.fixed:
                v = u.Quantity(parameter.value, parameter.unit)
            else:
                # units are lost in MCMC call, so they need to be recovered
                try:
                    v = u.Quantity(values[i], unit=parameter.unit)
                except u.core.UnitTypeError:
                    v = u.Dex(values[i], unit=parameter.unit)
                except u.core.UnitConversionError:
                    v = values[i] * parameter.unit
                i += 1
            current_parameters[name] = v
            self.parameters[name].value = v

        assert i == len(values), 'Not all parameters used.'

        return current_parameters

    def calculate_model_moments(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """
        Given a set of parameters, the method calculates the predicted first
        and second order moments of the target population.

        This method is a mere place-holder and should be overwritten by
        classes inheriting from `Runner()`.

        :param kwargs: This method does not accept any additional keyword
            arguments.
        :return: The first-order moments predicted for the target population.
        :return: The second-order moments predicted for the target population.
        """
        if kwargs:
            raise IOError('Unknown parameter(s) provided to calculate_model_moments: {0}'.format(kwargs.keys))
        return np.empty(self.n_data), np.empty(self.n_data)

    def lnprior(self, values: np.ndarray, parameters_to_ignore: list = None) -> float:
        """
        Checks if the priors on the parameters are fulfilled.

        :param values: The current values of the model parameters.
        :param parameters_to_ignore: In case the `fetch_parameters` method
            calculates any additional parameters not included in the
            Parameters() instance provided upon class initialization, those
            parameters should be provided as a list.

        :return: The negative log-likelihood corresponding to the priors. For
            uninformative priors, this is zero if the values are within their
            defined limits and -inf otherwise.
        """
        if parameters_to_ignore is None:
            parameters_to_ignore = []

        lnlike = 0
        for name, value in self.fetch_parameter_values(values).items():
            # check if parameter is valid
            if name not in self.parameters.keys():
                if name in parameters_to_ignore:
                    continue
                else:
                    raise IOError("Method 'lnprior()' received invalid parameter '{0}.".format(name))
            lnlike += self.parameters[name].evaluate_lnprior(value)
            if not np.isfinite(lnlike):
                return -np.inf
        return lnlike

    def lnlike(self, values: np.ndarray) -> float:
        """
        Calculates the likelihood of the model given the data without
        considering the priors.

        It is assumed that the distribution follows a Gaussian distribution.
        Therefore, the probability $p_i$ of a single measurement ($v_i$,
        $\\epsilon_{v,i}) is estimated as:

        $$
        p_i = \\exp(-(v_i - v_{\\rm SYS})^2 / (2(\\sigma^2 +
            \\epsilon_{{\\rm v},i}^2))) / (2(\\sigma^2 +
            \\epsilon_{{\\rm v}, i}^2))^{0.5}
        $$

        Then the log likelihood of the model is then determined by multiplying
        the probabilities of all measurements and taking the natural
        logarithm, resulting in

        $$
        \\ln L = \\sum_{i=0}^N -(v_i - v_{\\rm SYS})^2 / (2(\\sigma^2 +
            \\epsilon_{{\\rm v},i}^2)) + \\sum_{i=0}^N -0.5\\ln(2(\\sigma^2 +
            \\epsilon_{{\\rm v}, i}^2))
        $$

        In case that a background component is included, the $\\ln$
        probability of each measurement is given as

        $$
        \\ln p_i = \\ln(m_i p_{i, {\\rm pop}} + (1 - m_i) p_{i, {\\rm back}}).
        $$

        Here, $m_i$ is the membership prior of the star in question and
        $p_{i, {\\rm pop}}$ and $p_{i, {\\rm back}}$ are the probabilities
        derived for the population and background models, respectively.

        *Note* that when adding the probabilities of the population and
        background models, the exponential may result in an underflow. In
        order to avoid this, we follow the suggestion provided
        [here](https://stats.stackexchange.com/questions/142254/).

        :param values: The current values of the model parameters.

        :return: The negative log likelihood of the model given the data
            before taking into account the priors.
        """
        # Convert parameter values to dictionary
        parameters = self.fetch_parameter_values(values)

        # if needed split parameters into model and background components
        background_parameters = {}
        if self.background is not None:
            for bp in self.background_parameters:
                background_parameters[bp] = parameters.pop(bp)
            f_back = parameters.pop('f_back')
        else:
            f_back = 0

        v_target, sigma_target = self.calculate_model_moments(**parameters)

        # initial definitions to facilitate lnlike calculation
        norm = self.verr * self.verr + sigma_target * sigma_target
        exponent = -0.5*np.power(self.v - v_target, 2)/norm

        if self.background is None and self.lnlike_background is None:
            # in cases without a background population, the log likelihood is
            # SUM_{i=1}^N LN[EXP(-0.5*(v_i - v_los)^2/(verr_i + sigma_los)^2) / SQRT(2.*PI*(verr_i + sigma_los)^2)]
            # = SUM_{i=1}^{N} -0.5*(v_i - v_los)^2/(verr_i + sigma_los)^2 \
            #     + SUM_{i=1}^{N} -0.5*LN(2.*PI*(verr_i + sigma_los)^2)
            sum1 = -0.5 * np.sum(np.log(2. * np.pi * norm.value))
            sum2 = np.sum(exponent)
            return sum1 + sum2
        else:
            # in cases with a background population, the log likelihood per star can be written as the sum of two
            # exponential functions:
            # lnlike_i = LN(like_i)
            #            LN[pmember_i*EXP(lnlike_member_i) + (1 - pmember_i)*EXP(lnlike_background_i)]
            # For small likelihoods, the exponential functions may cause underflows. To avoid this, the log-sum-exp
            # trick is used. From both exponents their maximum value is subtracted and added as an additive term to
            # the final log likelihood.
            lnlike_member = -0.5*np.log(2. * np.pi * norm.value) + exponent

            if self.background is not None:
                v_back, sigma_back = self.background(**background_parameters)
                norm_back = self.verr * self.verr + sigma_back * sigma_back
                exponent_back = -0.5 * np.power(self.v - v_back, 2) / norm_back
                lnlike_background = -0.5*np.log(2. * np.pi * norm_back.value) + exponent_back
            else:
                lnlike_background = self.lnlike_background

            # get membership priors
            m = self.density / (self.density + f_back)

            max_lnlike = np.max([lnlike_member, lnlike_background], axis=0)
            lnlike = max_lnlike + np.log(m*np.exp(lnlike_member - max_lnlike) + (1. - m)*np.exp(
                lnlike_background - max_lnlike))

            return lnlike.sum()

    def lnprob(self, values: np.ndarray) -> float:
        """
        Calculates the likelihood of the current model including the priors.

        :param values: The current values of the model parameters.

        :return: The negative log likelihood of the model given the data after
            taking into account the priors.
        """
        lp = self.lnprior(values)
        if not np.isfinite(lp):
            return -np.inf
        return self.lnlike(values) + lp

    def get_initials(self, n_walkers: int) -> np.ndarray:
        """
        Create initial values for the MCMC chains.

        :param n_walkers: The number of walkers for which initial guesses
            should be created.

        :return: The initial guesses for the requested number of chains.
        """
        initials = np.zeros((n_walkers, self.n_fitted_parameters))
        i = 0
        for name, parameter in self.parameters.items():
            if parameter.fixed:
                continue
            else:
                initials[:, i] = parameter.evaluate_initials(n_walkers)
            i += 1
        return initials

    def __call__(self, n_walkers: int = 100, n_steps: int = 500, n_burn: int = 100, n_threads: int = 1,
                 n_out: int = None, pos: np.ndarray = None, lnprob0: np.ndarray = None, plot: bool = False,
                 prefix: str = 'sampler', true_values: list = None, **kwargs) -> emcee.EnsembleSampler:
        """
        Use an MCMC approach to determine the intrinsic parameters of the
        model for the velocity distribution.

        :param n_walkers: The number of walkers used for the MCMC analysis.
        :param n_steps: The number of steps that is performed by each walker.
        :param n_burn: Number of steps ignored at the start of each chain when
            getting results from chains.
        :param n_threads: The number of threads used to run the MCMC sampler.
        :param n_out: Number of steps after which the progress of the walkers
            will be saved.
        :param pos: Initial values for the walkers. If provided, must be a 2D
            array with shape (`n_walkers`, `n_parameters`).
        :param lnprob0: The list of log posterior probabilities for the
            walkers at positions given by pos. If lnprob is None, the initial
            values are calculated. It should have the shape (`n_walkers`,
            `n_parameters`).
        :param plot: Flag indicating if a plot showing the progress of the
            walkers at every n_out'th step should be created.
        :param prefix: Common prefix for names of output files produced by
            the code.
        :param true_values: A list containing the actual parameter values. If
            provided, each value will be displayed as a horizontal line when
            plotting the status of the chains. Only has an effect if the plot
            parameter is enabled.
        :return: The instance in which the MCMC walkers operate. Check the
            documentation of emcee.EnsembleSampler for further details.
        """
        if kwargs:
            if "filename" in kwargs or "plotfilename" in kwargs:
                logger.warning('Parameters <filename> and <plotfilename> not used anymore. Use <prefix> instead.')

        if plot:
            fig, _ = plt.subplots(self.n_fitted_parameters, 1, sharex='all', figsize=(8, 9))
        else:
            fig = None

        # define initial positions of the walkers in parameter space
        if pos is not None:
            assert pos.shape == (n_walkers, self.n_fitted_parameters), 'Array with starting values has invalid shape.'
        else:
            pos = self.get_initials(n_walkers=n_walkers)

        # check if starting values fulfil priors
        for i in range(n_walkers):
            if not np.isfinite(self.lnprior(pos[i])):
                raise ValueError(
                    "Invalid initial guesses for walker {0}: {1}={2}".format(i, self.fitted_parameters, pos[i]))

        # start MCMC
        if n_threads > 1:
            pool = Pool(processes=n_threads)
        else:
            pool = None

        sampler = emcee.EnsembleSampler(n_walkers, self.n_fitted_parameters, self.lnprob, pool=pool)
        logger.info("Running MCMC chain ...")

        if n_out is not None:
            msg = "Iter. <log like>   "
            i = 0
            for name, parameter in self.parameters.items():
                if not parameter.fixed:
                    msg += " {0:12s}".format('<' + name + '>')
                    i += 1
            logger.info(msg)

        state = None
        while sampler.iteration < n_steps:

            pos, lnp, state = sampler.run_mcmc(pos, n_out if n_out is not None else n_steps, log_prob0=lnprob0,
                                               rstate0=state, progress=True)

            if n_out is not None:
                output = " {0:4d} {1:12.5e}".format(sampler.iteration, np.mean(lnp[:]))
                i = 0
                for parameter in self.parameters.values():
                    if not parameter.fixed:
                        output += " {0:12.5e}".format(np.mean(pos[-n_out:-1, i]))
                        i += 1

                if sampler.iteration % n_out == 0:
                    if prefix is not None:
                        self.save_current_status(sampler, prefix=prefix)
                    if plot:
                        for ax in fig.axes:
                            ax.cla()
                        self.plot_chain(sampler.chain, true_values=true_values, figure=fig,
                                        filename='{0}_chains.png'.format(prefix) if prefix is not None else None)
                logger.info(output)

        if pool is not None:
            pool.close()

        # return current state of sampler
        return sampler

    @staticmethod
    def save_chain(sampler, filename="samplerchain.pkl"):
        # DEPRECATED
        warnings.warn('Method Runner.save_chain() is deprecated. Use Runner.save_current_status() instead.',
                      DeprecationWarning)

        prefix = filename.split('.')[0]
        if len(prefix) > 5 and prefix[-5:] == 'chain':
            prefix = prefix[:-5]

        Runner.save_current_status(sampler, prefix=prefix)

    @staticmethod
    def save_current_status(sampler: emcee.EnsembleSampler, prefix: str = "sampler"):
        """
        Save the current chain and the log-probabilities to two separate
        files after pickling them.

        :param sampler: The instance in which the MCMC walkers operate.
        :param prefix : The common name prefix of the files used to store the
            pickled chains and log-probabilities.
        """
        samples = sampler.chain
        lnprob = sampler.lnprobability

        with open('{0}_chain.pkl'.format(prefix), 'wb') as f:
            pickle.dump(samples, f)
        with open('{0}_lnprob.pkl'.format(prefix), 'wb') as f:
            pickle.dump(lnprob, f)

    @staticmethod
    def read_chain(filename: str = "samplerchain.pkl") -> np.ndarray:
        """
        Recover the complete chain from a previously pickled sampler.

        :param filename: The name of the file used to store the pickled
            sampler state.
        :return: The values sampled by the pickled chain.
        """
        file_object = open(filename, 'rb')
        return pickle.load(file_object)

    @staticmethod
    def read_final_chain(filename: str = "restart.plk") -> np.ndarray:
        """
        Recover the final step of the chain from a previously pickled sampler.

        :param filename : The name of the file used to store the pickled
            sampler state.
        :return: The values sampled by the pickled chain in the final step.
        """
        file_object = open(filename, 'rb')
        chain = pickle.load(file_object)
        file_object.close()

        last = chain[:, -1, :]
        return last

    def convert_to_parameters(self, chain: np.ndarray, n_burn: int) -> dict:
        """
        Convert the parameters values stored in the MCMC chain into a
        dictionary containing one entry per parameter.

        :param chain: The chain returned by the MCMC analysis
        :param n_burn: The number of steps at the beginning of the chain
            discarded as burn-in.

        :return: The model parameters sampled by the chain as a dictionary.
        """
        pars = {}
        n_samples = chain.shape[0]*(chain.shape[1] - n_burn)

        # first check: has the parameter been optimized?
        for par in self.parameters:
            if par in self.fitted_parameters:
                i = self.fitted_parameters.index(par)
                pars[par] = chain[:, n_burn:, i].flatten()

        # second check: was a fixed value used?
        for fix_par in [p for p in self.parameters if p not in pars]:
            if self.parameters[fix_par].expr is None:
                pars[fix_par] = np.full(n_samples, self.parameters[fix_par].value)

        # third check: is the parameter value determined using an expression?
        for dep_par in [p for p in self.parameters if p not in pars]:
            if self.parameters[dep_par].expr is not None:
                values = np.zeros(n_samples, dtype=np.float64)
                for n in range(n_samples):
                    for par in [p for p in pars.keys() if p in self.parameters[dep_par]._expr_deps]:
                        self.parameters[par].value = pars[par][n]
                        values[n] = self.parameters[dep_par].value
                pars[dep_par] = values

        return pars

    def compute_percentiles(self, chain: np.ndarray, n_burn: int, pct: list = None) -> np.ndarray:
        """
        Determine the requested percentiles of the parameter distributions
        obtained by the MCMC walkers.

        :param chain: The chains produced by the MCMC sampler. They should be
            provided as a 3D array, containing the parameters as first index,
            the steps as second index, and the walkers as third index.
        :param n_burn: The number of steps to be omitted from the calculation
            at the start of each MCMC walker.
        :param pct: The percentiles to be computed. The default is to compute
            the 16th, 50th, and 84th percentile for each parameter.
        :return: The requested percentiles for each of the fitted parameters.
        """
        if pct is None:
            pct = [16, 50, 84]

        _samples = chain[:, n_burn:, :].reshape((-1, self.n_fitted_parameters))

        # # check if position angle has been fitted; if so, needs special handling
        # if 'theta_0' not in self.fixed.keys():
        #
        #     i = self.fitted_parameters.index('theta_0')
        #
        #     # to avoid discontinuity at 2*PI, shift angles so that median angle lies at PI
        #     # for calculating median of an angle, see https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        #     offset = np.pi - np.arctan2(np.sin(_samples[:, i]).sum(), np.cos(_samples[:, i]).sum())
        #
        #     # also make sure angles are within range [0, 2*PI)
        #     _samples[:, i] = np.mod(_samples[:, i] + offset, 2. * np.pi)
        #
        #     results = np.percentile(_samples, pct, axis=0)
        #
        #     # undo shift
        #     results[:, i] -= offset
        #     return results
        #
        # else:
        return np.percentile(_samples, pct, axis=0)

    def compute_bestfit_values(self, chain: np.ndarray, n_burn: int) -> QTable:
        """
        Obtain estimates for the median values and the upper and lower limits
        of the uncertainty interval of each model parameter.

        This is done by first determining the 16%, 50%, and 86% percentiles
        from the distributions returned by the MCMC chains. The uncertainties
        are then estimated as p[86%] - p[50%]  and p[50%] - p[16%]

        :param chain: The chains produced by the MCMC sampler. They should be
            provided as a 3D array, containing the parameters as first index,
            the steps as second index, and the walkers as third index.
        :param n_burn: The number of steps to be omitted from the calculation
            at the start of each MCMC walker.

        :return: The median value and the upper and lower uncertainty for each
            of the fitted parameters. One column per parameter.
        """
        percentiles = self.compute_percentiles(chain, n_burn=n_burn, pct=[16, 50, 84])

        results = QTable(data=[['median', 'uperr', 'loerr']], names=['value'])
        results.add_index('value')

        i = 0
        for name, parameter in self.parameters.items():
            if parameter.fixed:
                continue
            else:
                parameter.value = percentiles[1, i]
                col = QTable.Column(
                    [percentiles[1, i], percentiles[2, i] - percentiles[1, i], percentiles[1, i] - percentiles[0, i]],
                    name=name, unit=parameter.unit)
                results.add_column(col)
                i += 1

        # mapper = map(lambda p: (p[1], p[2] - p[1], p[1] - p[0]), zip(*percentiles))
        # return [result for result in mapper]

        return results

    @property
    def labels(self) -> list:
        """
        :return: The labels used to indicate the fitted parameters in plots
            that illustrate the results/status of the sampler.
        """
        # recover parameters that have been fitted and their labels
        labels = []
        for name, parameter in self.parameters.items():
            if not parameter.fixed:
                labels.append(parameter.label)
        return labels

    def plot_chain(self, chain: np.ndarray, filename: str = 'chains.png', true_values: list = None,
                   figure: Figure = None, lnprob: np.ndarray = None, plot_median: bool = False) -> Figure:
        """
        Create a plot showing the current status of the MCMC chains.

        For each fitted parameter, a new subplot will be created showing the
        value of the parameter as a function of the step of each of the MCMC
        chains.

        :param chain: The chains produced by the MCMC sampler. They should be
            provided as a 3D array, containing the parameters as first index,
            the steps as second index, and the walkers as third index.
        :param filename: Filename used to store the final plot.
        :param true_values: A list containing the actual parameter values. If
            provided, each value will be displayed as a horizontal line.
        :param figure: The figure instance used as canvas of the plot. If
            provided, it must contain as may axes/subplots as free parameters.
        :param lnprob: Array containing the log probabilities of the models
            sampled by the chain. If provided, they are used to color-code the
            chains according to their likelihoods. Its shape must match that
            of the last two axes of the chain.
        :param plot_median: Overplot median of each distribution?
        :return: The figure instance prepared by the method.
        """
        # if figure instance provided, check that it comes with correct number of subplots
        if figure is not None:
            assert len(figure.axes) == self.n_fitted_parameters, 'No. of axes does not match no. of parameters.'
        else:
            figure = plt.figure(figsize=(8, 1 + 2*self.n_fitted_parameters))
            gs = gridspec.GridSpec(self.n_fitted_parameters, 1)
            ax_ref = None
            for i in range(self.n_fitted_parameters):
                ax = figure.add_subplot(gs[i], sharex=ax_ref)
                if not ax_ref:
                    ax_ref = ax
        axes = figure.axes

        samples = np.copy(chain)

        # make sure angles are within range [0, 2*PI)
        # if 'theta_0' in self.fitted_parameters:
        #     i = self.fitted_parameters.index('theta_0')
        #     samples[:, :, i] = np.mod(samples[:, :, i], 2. * np.pi)

        for i in range(self.n_fitted_parameters):
            if lnprob is None:
                axes[i].plot(samples[..., i].T, color="#AAAAAA", alpha=0.1)
            else:
                x, _ = np.mgrid[0:samples.shape[1]:1, 0:samples.shape[0]:1]
                xy = np.dstack((x, samples[..., i].T))
                _xy = xy.reshape(-1, 2)
                segments = np.concatenate([_xy[:-samples.shape[0], np.newaxis], _xy[samples.shape[0]:, np.newaxis]],
                                          axis=1)
                vmin, vmax = np.percentile(lnprob, [5, 95])
                norm = plt.Normalize(vmin, vmax)
                lc = LineCollection(segments, cmap='viridis', norm=norm)
                lc.set_array(lnprob[:, 1:].T.flatten())
                _ = axes[i].add_collection(lc)
            axes[i].set_ylim(samples[..., i].min(), samples[..., i].max())
            axes[i].yaxis.set_major_locator(MaxNLocator(5))
            
            if plot_median:
                axes[i].plot(np.percentile(samples[..., i].T, 16, axis=1), color='tab:red', alpha=1, lw=1.5)
                axes[i].plot(np.percentile(samples[..., i].T, 84, axis=1), color='tab:red', alpha=1, lw=1.5)
                axes[i].plot(np.median(samples[..., i].T, axis=1), color='tab:red', alpha=1, lw=1.5)
            
            if true_values is not None:
                axes[i].axhline(true_values[i], color="#888888", lw=2)
            axes[i].set_ylabel(self.labels[i])

            if i > 0:
                axes[i].set_xticklabels([])
            else:
                axes[i].set_xlim(0, samples.shape[1])

        figure.tight_layout(h_pad=0.0)
        if filename is not None:
            figure.savefig(filename)

        return figure

    def create_triangle_plot(self, chain: np.ndarray, n_burn: int, filename: str = 'corner.png', **kwargs) -> Figure:
        """
        Create a triangle plot showing 1D and 2D distributions of the values
        sampled by the MCMC walkers.

        :param chain: The chains produced by the MCMC sampler. They should be
            provided as a 3D array, containing the parameters as first index,
            the steps as second index, and the walkers as third index.
        :param n_burn: The number of steps that is neglected ('burned') at the
            beginning of each walker.
        :param filename: Filename used to store the final plot.
        :param kwargs: Any remaining keyword arguments are sent to the call of
            corner which is used to create the actual plot.
        :return: The Figure instance prepared by the method.
        """
        samples = np.copy(chain)[:, n_burn:, :].reshape((-1, self.n_fitted_parameters))

        # make sure angles are within range [0, 2*PI)
        # if 'theta_0' in self.fitted_parameters:
        #     i = self.fitted_parameters.index('theta_0')
        #     samples[:, i] = np.mod(samples[:, i], 2. * np.pi)

        if 'labels' not in kwargs.keys():
            kwargs['labels'] = self.labels
        corner_plot = corner.corner(samples,quantiles=[0.16, 0.5, 0.84],show_titles=True, **kwargs)

        axes = np.array(corner_plot.axes).reshape((self.n_fitted_parameters, self.n_fitted_parameters))
        for yi in range(1, self.n_fitted_parameters):
            ax = axes[yi, 0]
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
            # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            #     label.set_fontsize(14)
        for xi in range(self.n_fitted_parameters):
            ax = axes[-1, xi]
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
            # for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            #     label.set_fontsize(14)

        if filename is not None:
            corner_plot.savefig(filename)

        return corner_plot

    def sample_chain(self, chain: np.ndarray, n_burn: int, n_samples: int = 1) -> list:
        """
        Return the requested number of samples from the provided chain.

        :param chain: The chains produced by the MCMC sampler. They should be
            provided as a 3D array, containing the parameters as first index,
            the steps as second index, and the walkers as third index.
        :param n_burn: The number of steps that is neglected ('burned') at the
            beginning of each walker.
        :param n_samples: The number of samples that should be returned.
        :return: The parameter combinations randomly selected from the chain.
        """
        # select parameter sets randomly from provided chain
        _parameters = np.reshape(chain[:, n_burn:], (-1, chain.shape[-1]))
        indices = np.random.randint(0, _parameters.shape[0], (n_samples,))

        parameters = []
        for parameters_i in _parameters[indices]:
            parameters.append(self.fetch_parameter_values(parameters_i))

        return parameters
