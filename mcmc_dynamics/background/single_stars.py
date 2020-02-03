import logging
import numpy as np
from astropy import units as u


logger = logging.getLogger(__name__)


class SingleStars(object):
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
    def __init__(self, v):
        """
        Initialize a new instance of the SingleStar class.

        Parameters
        ----------
        v : array_like
            The radial velocities of the stars used to model the background
            population.
        """
        self.v = u.Quantity(v)
        if self.v.unit.is_unity():
            self.v *= u.km/u.s
            logger.warning('Missing units for <v> values. Assuming {0}.'.format(self.v.unit))
        self.n_stars = self.v.size

    def __call__(self, v, verr, sigma_int=0*u.km/u.s):
        """
        Calculates the log-likelihood for each provided velocity that is is
        part of the background population.

        Parameters
        ----------
        v : array_like
            The radial velocities for which the log-likelihoods are returned.
        verr : array_like
            The uncertainties taylored to the provided velocities. Array must
            have same shape as 'v'.
        sigma_int : float, optional
            Global scaling parameter for the widths of the Gaussian kernels.

        Returns
        -------
        lnlike : nd_array
            The log-likelihoods of the provided stars.
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

        lnlike = np.zeros(len(v), dtype=np.float64)

        for i in range(len(v)):
            norm = sigma_int**2 + verr[i]**2
            exp_coeff = -(self.v - v[i])**2/(2.*norm)
            exp_coeff_max = exp_coeff.max()

            lnlike[i] = exp_coeff.max() + np.log(np.sum(
                np.exp(exp_coeff - exp_coeff_max)/(np.sqrt(2.*np.pi*norm.value)))) - np.log(self.n_stars)

        return lnlike
