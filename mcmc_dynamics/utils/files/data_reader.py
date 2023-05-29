import logging
import numpy as np
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


    def compute_distances(self, ra_center, dec_center):
        """
        Calculates and returns distances of the data points relative to a
        given reference.

        Parameters
        ----------
        ra_center : instance of astropy.units.Quantity
            The right ascension of the reference point.
        dec_center : instance of astropy.unit.Quantity
            The declination of the reference point.

        Returns
        -------
        r : astropy.units.Quantity
            The distance of the data points relative to the reference.
        """
        if not self.has_coordinates:
            logger.error('Cannot calculate distances as world coordinates are missing.')
            return

        x, y = calc_xy_offset(self.data['ra'], self.data['dec'], ra_center, dec_center)
        return np.sqrt(x**2 + y**2)

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
