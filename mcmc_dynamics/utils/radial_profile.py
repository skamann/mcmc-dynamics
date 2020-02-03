import numpy as np


class RadialProfile(object):
    """
    Given a set of distances to a reference point, RadialProfile will sort the
    data into radial bins such that each bins contains a minimum number of
    stars and covers a minimum distance range in log-space.
    """

    def __init__(self, r):
        """
        Create a new instance of the RadialProfile class.

        Parameters
        ----------
        r : array_like
            The set of distances to the reference point.
        """
        self.r = np.asarray(r)
        self.n = self.r.shape[0]

        self.sorted_indices = np.argsort(self.r)
        self.r_sorted = self.r[self.sorted_indices]

    def __call__(self, nstars=50, dlogr=0.2):
        """
        Bin the data radially.

        Parameters
        ----------
        nstars : int, optional
            The minimum number of data points that must be in each bin.
        dlogr : float, optional
            The mininum radial range in log-space that each bin must cover,
            i.e. [log(r_max) - log(r_min)] >= dlogr

        Returns
        -------
        bin_number : ndarray, shape=(len(r), )
            The index of the bin that each data point was assigned to. Bin
            numbers increase with increasing distance. Data points that could
            not be assigned to any bin fulfilling the required properties are
            assigned -1.
        """
        bin_number = -np.ones(self.n, dtype=np.int16)

        i = 0
        while i < (self.n - nstars):

            j = min(self.n, i + nstars)

            while (np.log10(self.r_sorted[j]) - np.log10(self.r_sorted[i])) < dlogr:
                j += 1
                if j >= self.n:
                    break

            bin_number[i:j] = np.max(bin_number) + 1
            i = j

        return bin_number[self.sorted_indices.argsort()]
