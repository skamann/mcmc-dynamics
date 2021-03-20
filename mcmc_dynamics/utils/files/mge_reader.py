import logging
import numpy as np
import warnings
from astropy import units as u
from astropy.table import QTable
from astropy import table


logger = logging.getLogger(__name__)


class MgeReader(object):

    def __init__(self, data, lum=False, **kwargs):

        self.data = QTable(data, **kwargs)

        if lum:
            i_unit = u.solLum / u.pc**2
        else:
            i_unit = u.solMass / u.pc**2

        # we require four columns:
        #     n - component number,
        #     i - central intensity,
        #     s - sigma,
        #     q - axis ratio
        # in addition, if units are provided, they must be convertible:
        #     i - M_sun/pc^2 (or L_sun/pc^2 if lum=True),
        #     s - arcsec
        # if no units are provided, it is assumed that i and s are provided in the units listed above
        for required in ['i', 's']:
            assert required in self.data.columns, 'Missing required column {0} in input data.'.format(required)

        if self.data['i'].unit:
            try:
                _ = self.data['i'].unit.to(i_unit)
            except u.UnitConversionError as msg:
                logger.error('Provided units for column i are invalid: {0}'.format(msg))
        else:
            self.data['i'].unit = i_unit

        if self.data['s'].unit:
            try:
                _ = self.data['s'].unit.to(u.arcsec)
            except u.UnitConversionError as msg:
                logger.error('Provided units for column s are invalid: {0}'.format(msg))
        else:
            self.data['s'].unit = u.arcsec

        if 'n' not in self.data.columns:
            logger.warning('Input data misses column n. Assuming ascending component indices')
            self.data['n'] = np.arange(1, len(self.data) + 1)

        if 'q' not in self.data.columns:
            logger.warning('Input data misses column q. Assuming circularity (q=1).')
            self.data['q'] = 1.

    @property
    def n_components(self):
        """
        Returns the number of Gaussian components in the instance.
        """
        return len(self.data)

    def add_ellipticity(self, q):
        """
        For circular Gaussians, this method can be used to assign a global
        ellipticity.

        This is done under the assumption that the current widths of the
        Gaussians corresponds to the average radius of the ellipse, i.e.
        s=SQRT(q)*a, with the axis ratio q and the semi-major axis width a.
        The current widths will be set to the widths along the semi-major axes
        of the Gaussians.

        The intensities of the Gaussians are not affected by this
        transformation if the existing widths have been used to convert the
        integrated fluxes to intensities. Note that the MGE code assumes that
        the intensity is related to the integrated flux by i = f/(2*PI*q*a^2),
        and q*a^2=r^2.

        Parameters
        ----------
        q : float
             The axis ratio that should be assigned to the Gaussians. Must be
             in interval (0, 1].
        """
        if (self.data['q'] < 0).any():
            logger.error('Can only set axis ratios for circular Gaussians.')
            return

        self.data['q'] = q
        self.data['s'] /= np.sqrt(q)

    def eval(self, x, y, n=None):
        """
        Evaluate the MGE at the provided position(s).

        Parameters
        ----------
        x : array_like
            The x (=semi-major axis) coordinates of the positions where the
            MGE is evaluated.
        y : array_like
            The y (=semi-minor axis) coordinates of the positions where the
            MGE is evaluated.
        n : array_like, optional
            The components of the MGE to be used. By default, all are used.

        Returns
        -------
        mge : nd_array
            The value(s) of the MGE at the input position(s).
        """
        if n is None:
            n = self.data['n']
        assert np.isin(n, self.data['n']).all(), 'Invalid MGE components provided.'

        mge = np.zeros(x.shape)*self.data['i'].unit

        for row in self.data:
            mge += row['i']*np.exp((x**2 + y**2/row['q']**2)/(-2.*row['s']**2))

        return mge


def get_mge(filename):
    # read in the tracer density MGE from the example and add appropriate units
    _mge = table.Table.read(filename, format='ascii.ecsv')

    # if no axis ratios are provided, assume circularity and trigger warning
    if "q" not in _mge.columns:
        warnings.warn("No axis ratios provided for MGE components in '{0}'. Assuming q=1.".format(filename))
        _mge['q'] = 1.0

    mge_lum = MgeReader(_mge, lum=True)

    _mge['i'] = _mge['i'] * u.solMass / u.solLum
    mge_mass = MgeReader(_mge, lum=False)

    return mge_lum, mge_mass
    
    
def _get_dist(x, y, _x, _y):
    return np.sqrt((x - _x)**2 + (y - _y)**2)


def get_nearest_neigbhbour_idx(x, y, coords):
    dists = np.asarray([_get_dist(x, y, _x, _y,) for (_x, _y) in coords])
    return dists.argmin()


def get_nearest_neigbhbour_idx2(x, y, coords_dict):
    dists = {}
    for offset, filename in coords_dict.items():
        dists[offset] = _get_dist(x, y, offset[0], offset[1])
    min_coord = min(dists, key=dists.get)
    # min_coord = [k for k, v in dists.items() if v == min_dist][0]
    
    return min_coord
