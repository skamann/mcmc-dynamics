import logging
import numpy as np
from astropy import units as u


logger = logging.getLogger(__name__)


class Gaussian(object):

    def __init__(self, mean, sigma):

        self.mean = u.Quantity(mean)
        if self.mean.unit.is_unity():
            self.mean *= u.km/u.s
            logger.warning('Missing units for parameter <mean>. Assuming {0}.'.format(self.mean.unit))

        self.sigma = u.Quantity(sigma)
        if self.sigma.unit.is_unity():
            self.sigma *= u.km/u.s
            logger.warning('Missing units for parameter <sigma>. Assuming {0}.'.format(self.sigma.unit))

    def __call__(self, v, verr):

        norm = verr * verr + self.sigma * self.sigma
        exponent = -0.5 * np.power(v - self.mean, 2) / norm

        return -0.5 * np.log(2. * np.pi * norm.value) + exponent
