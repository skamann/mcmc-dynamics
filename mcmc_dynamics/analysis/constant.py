import inspect
import logging
import numpy as np
try:
    from importlib.resources import files
except ImportError:  # for Python v<3.9
    from importlib_resources import files
from astropy import units as u
from astropy.table import QTable

from .runner import Runner
from .. import config
from ..parameter import Parameters
from ..utils.coordinates import calc_xy_offset, get_amplitude_and_angle
from ..utils.files import DataReader

logger = logging.getLogger(__name__)


class ConstantFit(Runner):
    """
    The purpose of the ModelFit class is to fit the kinematics of a stellar
    population using a constant rotation field and velocity dispersion.

    Rotation is implemented by varying the systemic velocity $v_{\\rm SYS}$ as
    a function of position angle $\\theta$ as follows.

    $$
    v_{\\rm LOS}(\\theta) = v_{\\rm SYS} + v_{\\rm max}\\sin(
        \\theta - \\theta_{0}).
    $$

    This implies that $\\theta_{0}$ points along the rotation axis, and that
    the maximum velocity is found at $\\theta_{0} + \\pi/2$.

    Position angles are determined using [numpy.arctan2(dy, dx)](https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html#numpy-arctan2),
    with `dx` and `dy` representing offsets from the assumed centre of
    symmetry, calculated from the world coordinates of the data using the
    function `calc_xy_offsets`.


    As fitting angles can be difficult due to the discontinuity at $2\\pi$,
    instead of $\\theta_0$, the $x$ and $y$ components of $v_{\\rm MAX}$ are
    considered free parameters. Hence, the model has the following 7
    parameters that can be optimized. `v_sys`, `v_maxx`, `v_maxy`,
    `sigma_max`, `ra_center`, `dec_center`.
    """
    MODEL_PARAMETERS = ['v_sys', 'sigma_max', 'v_maxx', 'v_maxy', 'ra_center', 'dec_center']
    OBSERVABLES = {'v': u.km / u.s, 'verr': u.km / u.s, 'ra': u.deg, 'dec': u.deg}

    parameters_file = files(config).joinpath('constant.json')

    def __init__(self, data: DataReader, parameters: Parameters = None, **kwargs):
        """
        :param data: The observed data for a set of n stars. The instance must
            provide at least the RA and Dec coordinates, velocities and their
            uncertainties.
        :param parameters: The model parameters.
        :param kwargs: Any extra keyword arguments are forwarded to the
            initialization of the parent class.
        """
        # required observables
        self.ra = None
        self.dec = None

        if parameters is None:
            parameters = Parameters().load(self.parameters_file)

        super(ConstantFit, self).__init__(data=data, parameters=parameters, **kwargs)

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters
        self.dispersion_parameters = inspect.signature(self.dispersion_model).parameters

    def dispersion_model(self, sigma_max: float, **kwargs) -> np.ndarray:
        """
        Calculate the velocity dispersion at the positions of the available
        data points.

        :param sigma_max: The velocity dispersion is assumed to be constant
            in this model.
        :param kwargs: This method does not use any additional keyword
            arguments.
        :return: The model values of the velocity dispersion at the positions
            of the individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.dispersion_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        return sigma_max * np.ones(self.n_data, dtype=np.float64)

    def rotation_model(self, v_sys: u.Quantity, v_maxx: u.Quantity, v_maxy: u.Quantity, ra_center: u.Quantity,
                       dec_center: u.Quantity, **kwargs) -> np.ndarray:
        """
        Calculate the rotation velocity at the positions of the available data
        points.

        :param v_sys: The constant systemic velocity of the model.
        :param v_maxx: The x-component of the constant rotation velocity of
            the model.
        :param v_maxy: The y-component of the constant rotation velocity of
            the model.
        :param ra_center: The right ascension of the assumed center.
        :param dec_center: The declination of the assumed center.
        :param kwargs: This model does not use any additional keyword
            arguments.
        :return: The values of the rotation velocity at the positions of the
             individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.rotation_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        dx, dy = calc_xy_offset(ra=self.ra, dec=self.dec, ra_center=ra_center, dec_center=dec_center)
        theta = np.arctan2(dy, dx)

        v_max = np.sqrt(v_maxx ** 2 + v_maxy ** 2)
        theta_0 = np.arctan2(v_maxy, v_maxx)
        return v_sys + v_max * np.sin(theta - theta_0)

    def calculate_model_moments(self, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the first- and second-order moments predicted by the
        model for a given set of parameters

        :param kwargs: The values of the model parameters. Note that only the
            free parameters should be provided, in the order specified by the
            instance of `Parameters`.
        :return: The mean velocity predicted by the model at the positions of
            the observations.
        :return: The velocity dispersion predicted by the model at the
            positions of the observations.
        """
        # Collect parameters for method calls to evaluate rotation and dispersion models.
        kwargs_rotation = {}
        kwargs_dispersion = {}
        for parameter, value in kwargs.items():
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

        return v_los, sigma_los

    def compute_theta_vmax(self, chain: np.ndarray, n_burn: int,
                           return_samples: bool = False) -> tuple[QTable, np.ndarray, np.ndarray, np.ndarray]:
        """Computes the 16th, 50th, and 84th percentiles of the position angle
        `theta_0` and rotation amplitude `v_max` distributions.

        :param chain: The array containing the parameter values sampled during
            the MCMC analysis.
        :param n_burn: The number of steps considered as burn-in that are
            discarded when analysing the chain.
        :param return_samples: Whether the raw values taken from the chain
            should be returned in addition to their percentiles.
        :return: The table contains the parameters as columns and the
            percentiles as rows.
        :return: The individual `v_max` values, if `return_samples=True`.
        :return: The individual `theta_0` values, if `return_samples=True`.
        :return: The individual `sigma` values, if `return_samples=True`.
        """
        pars = self.convert_to_parameters(chain=chain, n_burn=n_burn)

        results, v_max, _theta = get_amplitude_and_angle(pars, return_samples=return_samples)

        if results is None:
            logger.error('Could not recover parameters of rotation field in {}.compute_theta_vmax().'.format(
                self.__class__.__name__))
            return QTable(), np.array([]), np.array([]), np.array([])
        else:
            results['v_max'] *= self.units['v_maxx']

        if return_samples:
            return results, v_max, _theta, pars['sigma']
        else:
            return results, np.array([]), np.array([]), np.array([])

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
