import logging
import numpy as np
from astropy import units as u
from astropy.table import QTable
from ..coordinates import calc_xy_offset


logger = logging.getLogger(__name__)


class DataReader(object):

    def __init__(self, data, **kwargs):
        """
        Initialize a new instance of the DataReader class.

        Parameters
        ----------
        data : numpy ndarray, dict, list, Table, or table-like object, optional
            Data to initialize the instance.
        kwargs
            Any additional arguments are passed on to the initialization of
            astropy.table.Table which is used to process the input data.

        Returns
        -------
        The newly created instance.
        """
        self.data = QTable(data, **kwargs)

    @property
    def sample_size(self):
        return len(self.data)

    @property
    def has_ra(self):
        return 'ra' in self.data.columns

    @property
    def has_dec(self):
        return 'dec' in self.data.columns

    @property
    def has_coordinates(self):
        return self.has_ra & self.has_dec

    # def rotate(self, alpha):
    #     """
    #     Rotate the coordinate system by an angle alpha around its origin.
    #
    #     Parameters
    #     ----------
    #     alpha : float
    #         The angle by which to rotate in counterclockwise direction.
    #
    #     Returns
    #     -------
    #     rotated_data : DataReader
    #        A new instance of the DataReader class is returned. The coordinates
    #        (if any) are transformed into the new coordinate system.
    #     """
    #     alpha = u.Quantity(alpha)
    #     if alpha.unit.is_unity():
    #         alpha *= u.rad
    #         logger.warning('Missing unit of parameter <alpha>. Assuming {0}.'.format(alpha.unit))
    #
    #     rotated_data = self.__class__(self.data)
    #     if not self.has_cartesian and not self.has_polar:
    #         logger.warning('Current table lacking coordinates to apply rotation to.')
    #     else:
    #         if self.has_cartesian:
    #             rotated_data.data['x'] = self.data['x']*np.cos(alpha) + self.data['y']*np.sin(alpha)
    #             rotated_data.data['y'] = -self.data['x']*np.sin(alpha) + self.data['y']*np.cos(alpha)
    #         if self.has_polar:
    #             rotated_data.data['theta'] -= alpha
    #
    #     return rotated_data
    #
    # def compute_polar(self):
    #     """
    #     Calculates polar coordinates from the cartesian ones and adds them to
    #     the data of the current instance.
    #     """
    #
    #     if not self.has_cartesian:
    #         logger.error('Cannot calculate polar coordinates as cartesian coordinates are missing.')
    #         return
    #
    #     self.data['r'] = np.sqrt(self.data['x']**2 + self.data['y']**2)
    #     # if self.data['x'].unit is not None:
    #     #     print(self.data['r'].unit)
    #     #     self.data['r'].unit = self.data['x'].unit
    #     self.data['theta'] = np.arctan2(self.data['y'], self.data['x'])
    #     # self.data['theta'].unit = u.rad
    #
    # def compute_cartesian(self):
    #     """
    #     Calculates cartesian coordinates from the polar ones and adds them to
    #     the data of the current instance.
    #     """
    #     if not self.has_polar:
    #         logger.error('Cannot calculate cartesian coordinates as polar coordinates are missing.')
    #         return
    #
    #     self.data['x'] = self.data['r']*np.cos(self.data['theta'])
    #     self.data['y'] = self.data['r']*np.sin(self.data['theta'])
    #     #if self.data['r'].unit is not None:
    #     #    print(self.data['x'])
    #     #    self.data['x'].unit = self.data['r'].unit
    #     #    self.data['y'].unit = self.data['r'].unit
    #
    # def apply_offset(self, x=0, y=0):
    #     """
    #     Subtracts the given values from all x- and y-coordinates.
    #     """
    #     self.data['x'] -= x
    #     self.data['y'] -= y

    def make_radial_bins(self, ra_center, dec_center, nstars=50, dlogr=0.2):
        """
        Create radial bins relative to the provided center.

        Parameters
        ----------
        ra_center : instance of astropy.units.Quantity
            The right ascension of the center around which to create the
            radial bins.
        dec_center : instance of astropy.units.Quantity
            The declination of the center around which to create the radial
            bins.
        nstars : int, optional
            The minimum number of stars per bin.
        dlogr : float, optional
            The minimum extent in log10(radius) that each bin covers.
        force : bool, optional
            Flag indicating if the radial bins should be determined
        """
        if not self.has_coordinates:
            logger.error('Cannot create radial profile. WCS coordinates of data points unknown.')
            return

        dx, dy = calc_xy_offset(ra=self.data['ra'], dec=self.data['dec'], ra_center=ra_center, dec_center=dec_center)
        r = np.sqrt(dx**2 + dy**2)

        sorted_indices = np.argsort(r)
        r_sorted = r[sorted_indices].value

        bin_number = -np.ones(self.sample_size, dtype=np.int16)

        i = 0
        while i < (self.sample_size - nstars):

            j = min(self.sample_size, i + nstars)

            while (np.log10(r_sorted[j]) - np.log10(r_sorted[i])) < dlogr:
                j += 1
                if j >= self.sample_size:
                    break

            bin_number[i:j] = np.max(bin_number) + 1
            i = j

        if (self.sample_size - i) > 0.5 * nstars or np.max(bin_number) == -1:
            bin_number[i:] = np.max(bin_number) + 1
        else:
            bin_number[i:] = np.max(bin_number)

        self.data['bin'] = bin_number[sorted_indices.argsort()]

    def fetch_radial_bin(self, i):
        """

        Parameters
        ----------
        i

        Returns
        -------

        """
        if 'bin' not in self.data.columns:
            logger.error('No information about bins available.')
            return None
        elif i < self.data['bin'].min() or i > self.data['bin'].max():
            logger.error('Requested bin {0} does not exist.'.format(i))
            return None

        return self.__class__(self.data[self.data['bin'] == i])
