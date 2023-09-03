import logging
import numpy as np
from astropy import units as u
try:
    from importlib.resources import files
except ImportError:  # for Python v<3.9
    from importlib_resources import files

from .background import Background
from .. import config

logger = logging.getLogger(__name__)


class SingleStars(Background):
    """
    Set up a background population consisting of M individual stars that are
    only characterized by their radial velocities.

    The likelihood that a test star i with velocity v_i and uncertainty verr_i
    is part of the background population is approximated as the superposition
    of Gaussian kernels:

    p(v_i, verr_i) = (1/M)*SUM_{j=1}^{M} [ \
        EXP(-0.5*(v_i - v_j)^2/(verr_i^2 + sigma_int^2) / SQRT( \
            2.*PI*(verr_i^2 + sigma_int^2))
            )
        ],

    where sigma_int is a global scaling parameter that is defaulted to zero.
    """
    parameters_file = files(config).joinpath('single_stars_background.json')

    def __init__(self, v: u.Quantity, n_stars: int = None):
        """
        :param v: The radial velocities of the stars used to model the
            background population.
        :param n_stars: The number of stars for which the background is
            modeled.
        """
        super(SingleStars, self).__init__(n_stars=n_stars)

        self.v = u.Quantity(v)
        if self.v.unit.is_unity():
            self.v *= u.km/u.s
            logger.warning('Missing units for <v> values. Assuming {0}.'.format(self.v.unit))
        self.n_model = self.v.size
        self.n_stars = n_stars

    def lnlike(self, v: np.ndarray, verr: np.ndarray, sigma_int: u.Quantity = 0*u.km/u.s) -> np.ndarray:
        """
        Calculates the log-likelihood for each provided velocity that it is
        part of the background population.

        :param v: The radial velocities for which the log-likelihoods are
            returned.
        :param verr: The uncertainties tailored to the provided velocities.
            Array must have same shape as 'v'.
        :param sigma_int: Global scaling parameter for the widths of the
            Gaussian kernels.
        :return: The log-likelihoods of the provided stars.
        """
        # Important: when calculating the log-likelihoods, the exponents of the exp-functions in the Gaussian kernels
        # can be very small. To avoid under/-overflowing, the log-sum-exp trick is used. it works by subtracting from
        # all exponents in the sum their maximum value and adding it again to the final log-likelihood. See
        # https://stats.stackexchange.com/questions/142254/ \
        #     what-to-do-when-your-likelihood-function-has-a-double-product-with-small-values
        sigma_int = u.Quantity(sigma_int)
        if sigma_int.unit.is_unity():
            sigma_int *= u.km/u.s
            logger.warning('Missing quantity for parameter <sigma_int>. Assuming {0}.'.format(sigma_int.unit))

        norm = sigma_int**2 + verr**2
        exp_coeff = -(np.subtract.outer(self.v, v)) ** 2 / (2. * norm)
        exp_coeff_max = np.max(exp_coeff, axis=0)
        lnlike = exp_coeff_max + np.log(np.sum(np.exp(exp_coeff - exp_coeff_max)/
                                               (np.sqrt(2.*np.pi*norm.value)), axis=0)) - np.log(self.n_model)
        return lnlike

    def __call__(self, sigma_int: float) -> tuple[np.ndarray, np.ndarray]:

        raise NotImplementedError
