import logging
import numpy as np
try:
    from importlib.resources import files
except ImportError:  # for Python v<3.9
    from importlib_resources import files

from .background import Background
from .. import config


logger = logging.getLogger(__name__)


class Gaussian(Background):
    """
    Use a Gaussian velocity distribution to model the background.
    """
    parameters_file = files(config).joinpath('gaussian_background.json')

    @staticmethod
    def lnlike(v: np.ndarray, verr: np.ndarray, v_back: float, sigma_back: float) -> np.ndarray:
        """
        Calculate the log-likelihood for a fixed set of model parameters and
        observed velocities.

        :param v: The observed velocities.
        :param verr: The uncertainties of the observed velocities
        :param v_back: The mean of the Gaussian velocity distribution model.
        :param sigma_back: The standard deviation of the Gaussian velocity
            distribution model.
        :return: The log-likelihoods for the provided velocity observations.
        """

        norm = verr * verr + sigma_back * sigma_back
        exponent = -0.5 * np.power(v - v_back, 2) / norm

        return -0.5 * np.log(2. * np.pi * norm.value) + exponent

    def __call__(self, v_back: float, sigma_back: float, **kwargs) -> tuple[float, float]:
        """
        This method can be provided as `background` parameter when
        initializing a new instance of `Runner()`.

        The method takes the current values of the background parameters as
        input and uses them to compute the first and second order moments
        of the background velocity distribution at the locations of the stars.
        In this case with constant mean and dispersion, input and output
        values are the same.

        :param v_back: The mean of the Gaussian velocity distribution model.
        :param sigma_back: The standard deviation of the Gaussian velocity
            distribution model.
        :param kwargs: This method does not accept any additional keyword
            arguments
        :return: The mean of the Gaussian velocity distribution model at the
            locations of the stars.
        :return: The standard deviation of the Gaussian velocity distribution
            model at the locations of the stars.
        """
        if kwargs:
            raise IOError(f'Unknown parameters {kwargs} provided when evaluating Gaussian() background.')
        return v_back, sigma_back
