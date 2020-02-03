import logging
import numpy as np
from astropy import units as u
from scipy import integrate


logger = logging.getLogger(__name__)


class VSigma(object):

    def __init__(self, r, density, v_max, sigma):

        self.r = u.Quantity(r)
        if self.r.unit.is_unity():
            self.r *= u.arcmin
            logger.warning('Missing unit of parameter <r>. Assuming {0}.'.format(self.r.unit))

        self.density = u.Quantity(density)
        if self.density.unit.is_unity():
            self.density /= u.arcmin**2
            logger.warning('Missing unit of parameter <density>. Assuming {0}.'.format(self.density.unit))

        self.v_max = u.Quantity(v_max)
        if self.v_max.unit.is_unity():
            self.density *= u.km/u.s
            logger.warning('Missing unit of parameter <v_max>. Assuming {0}.'.format(self.v_max.unit))

        self.sigma = u.Quantity(sigma)
        if self.sigma.unit.is_unity():
            self.sigma *= u.km/u.s
            logger.warning('Missing unit of parameter <sigma>. Assuming {0}.'.format(self.sigma.unit))

    def __call__(self, r_outer):

        r_outer = u.Quantity(r_outer)
        if r_outer.unit.is_unity():
            r_outer *= u.arcmin
            logger.warning('Missing unit of parameter <r>. Assuming {0}.'.format(r_outer.unit))
        r_outer = r_outer.to(self.r.unit)

        if r_outer > self.r.max():
            logger.error('Provided radius for calculating V/Sigma outside data range.')
            return np.nan

        slc = (self.r < r_outer)
        r = np.append(self.r[slc], r_outer).value * self.r.unit
        density = np.append(self.density[slc], np.interp(r_outer, self.r, self.density))
        v_max = np.append(self.v_max[slc], np.interp(r_outer, self.r, self.v_max))
        sigma = np.append(self.sigma[slc], np.interp(r_outer, self.r, self.sigma))

        vsigma2 = integrate.simps(density*0.5*v_max**2*r, r) / integrate.simps(density*sigma**2*r, r)

        lambdar = integrate.simps(r**2*density*(2./np.pi)*v_max) / integrate.simps(
            r**2*density*np.sqrt(sigma**2 + 0.5*v_max**2))

        return np.sqrt(vsigma2), lambdar
