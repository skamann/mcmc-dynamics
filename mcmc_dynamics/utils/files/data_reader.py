import logging
import numpy as np
from astropy import units as u
from astropy.table import QTable
from ..coordinates import calc_xy_offset


logger = logging.getLogger(__name__)


class DataReader(object):
    """
    DataReader provides a framework for reading and processing a set of
    observed data.
    """

    def __init__(self, data, **kwargs):
        """
        Initialize a new instance of the DataReader class.

        :param data: The data used to initialize the instance. Can be any
            format compatible with an astropy QTable.
        :param kwargs: Any additional arguments are passed on to the
            initialization of astropy.table.QTable which is used to process
            the input data.
        """
        self.data = QTable(data, **kwargs)

    @property
    def sample_size(self) -> int:
        """
        :return: The number of measurements.
        """
        return len(self.data)

    @property
    def has_ra(self) -> bool:
        """
        :return: Flag indicating if right ascension coordinates are available.
        """
        return 'ra' in self.data.columns

    @property
    def has_dec(self) -> bool:
        """
        :return: Flag indicating if declination coordinates are available.
        """
        return 'dec' in self.data.columns

    @property
    def has_coordinates(self) -> bool:
        """
        :return: Flag indicating if WCS coordinates are available.
        """
        return self.has_ra & self.has_dec

    def compute_distances(self, ra_center: u.Quantity, dec_center: u.Quantity) -> u.Quantity:
        """
        Calculates and returns distances of the data points relative to a
        given reference.

        :param ra_center: The right ascension of the reference point.
        :param dec_center: The declination of the reference point.
        :return: The distance of the data points relative to the reference.
        :raises IOError: If the instance lacks WCS coordinates.
        """
        if not self.has_coordinates:
            raise IOError('Cannot calculate distances as world coordinates are missing.')

        x, y = calc_xy_offset(self.data['ra'], self.data['dec'], ra_center, dec_center)
        return np.sqrt(x**2 + y**2)

    def make_radial_bins(self, ra_center: u.Quantity, dec_center: u.Quantity, nstars: int = 50,
                         dlogr: float = 0.2):
        """
        Create radial bins relative to the provided center.

        :param ra_center: The right ascension of the center around which to
            create the radial bins.
        :param dec_center: The declination of the center around which to
            create the radial bins.
        :param nstars: The minimum number of stars per bin.
        :param dlogr: The minimum extent in log10(radius) that each bin covers.
        """
        r = self.compute_distances(ra_center=ra_center, dec_center=dec_center)

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

    def fetch_radial_bin(self, i: int):
        """
        Fetch the data for an individual radial bin.

        :param i: The index of the radial bin requested.
        :return: The data for the stars in the radial bin.
        :raises NotImplementedError: If no bins have been determined.
        :raises IndexError: if requested bin does not exist.
        """
        if 'bin' not in self.data.columns:
            raise NotImplementedError('No information about bins available.')
        elif i < self.data['bin'].min() or i > self.data['bin'].max():
            raise IndexError('Bin indices range from {} to {}, got {}'.format(
                self.data['bin'].min(), self.data['bin'].max(), i))
        return self.__class__(self.data[self.data['bin'] == i])
