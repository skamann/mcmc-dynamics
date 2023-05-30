import numpy as np
from scipy.spatial import ConvexHull
from astropy import units as u
from astropy.table import QTable


def get_eccentricity_and_pa(dx, dy, radii=None, bootstrap=False, seed=None):
    """
    Given a distribution of (x, y) coordinates, the function will determine
    the eccentricity and the position angle of the semi-major axis of the
    input distribution for a given set of annuli.

    The shape parameters are calculated using the covariance matrix between
    the provided x- and y-coordinates. The eigenvectors of this matrix
    correspond to the semi-major and semi-minor axes of the distribution, so
    that eccentricity and position angle can be readily computed.

    Parameters
    ----------
    dx : array_like
        The x-coordinates of the input distribution.
    dy : array_like
        The y-coordinates of the input distribution.
    radii : array_like, optional
        The inner and outer radii of the annuli for which the eccentricity
        and position angle are measured.
    bootstrap : bool, optional
        Flag indicating if the uncertainties of the shape parameters should
        be estimated using a bootstrap method.
    seed : int, optional
        Seed used to initialize the random number generator for the bootstrap
        method.

    Returns
    -------
    results : instance of astropy.table.QTable
        The table will contain one row per requested annulus, containing the
        limiting radii (`r_min`, `r_max`), the mean radius `r_mean`, the
        number of sources entering the calculation `n`, the eccentricity `e`,
        the position angle of the semi-major axis `theta`, and the fraction of
        the annulus that is covered by the data `frac`. In case uncertainties
        are requested, they are stored in columns `e_err` and `theta_err`.
    """
    rng = np.random.default_rng(seed=seed)

    # make sure input is available as astropy Quantity objects so that unit
    # information can be accessed and stored.
    if not isinstance(dx, u.Quantity):
        dx = u.Quantity(dx)
    if not isinstance(dy, u.Quantity):
        dy = u.Quantity(dy)

    dr = np.sqrt(dx**2 + dy**2)

    # get convex hull around photometry to estimate fraction of annuli covered by data
    hull = ConvexHull(np.stack((dx, dy), axis=1))

    # if no radii are provided, determine shape parameters in one circle with a radius corresponding to
    # approximately half the edge length of the input data
    if radii is None:
        radii = [0, 0.5*np.sqrt(hull.volume)*dr.unit]
    if len(radii)  == 1:
        radii = np.r_[0, radii]

    # prepare output array
    results = QTable(data=np.zeros((len(radii)-1, 9), dtype=np.float64),
                     names=['r_min', 'r_max', 'r_mean', 'n', 'e', 'e_err', 'theta', 'theta_err', 'frac'],
                     units=[dr.unit, dr.unit, dr.unit, None, None, None, u.rad, u.rad, None])

    # calculate covariance matrix
    for ii, r_min in enumerate(radii[:-1]):
        r_max = radii[ii + 1]

        slc = (dr >= r_min) & (dr < r_max)

        n = slc.sum()
        cov = np.array([[np.sum(dx[slc] * dx[slc]).value / n, np.sum(dx[slc] * dy[slc]).value / n],
                        [np.sum(dy[slc] * dx[slc]).value / n, np.sum(dy[slc] * dy[slc]).value / n]])

        w, v = np.linalg.eig(cov)
        i = w.argmax()
        j = w.argmin()

        # make sure zeropoint of position angle is y- instead of x-axis
        theta = np.arctan2(v[1, i], v[0, i]) - np.pi / 2.
        if theta < -np.pi:
            theta += 2. * np.pi
        e = np.sqrt(1. - w[j]**2 / w[i]**2)

        results[ii][['r_min', 'r_max', 'r_mean', 'n', 'e', 'theta']] = [
            r_min, r_max, np.mean(dr[slc]), n, e, theta*u.rad]

        if bootstrap:
            theta_samples = []
            e_samples = []
            for _ in range(100):
                random_indices = np.flatnonzero(slc)[rng.integers(0, n, size=(n,))]
                _dx = dx[random_indices]
                _dy = dy[random_indices]

                _cov = np.array([[np.sum(_dx * _dx).value / n, np.sum(_dx * _dy).value / n],
                                 [np.sum(_dy * _dx).value / n, np.sum(_dy * _dy).value / n]])

                _w, _v = np.linalg.eig(_cov)
                _i = _w.argmax()
                _j = _w.argmin()

                theta_samples.append(np.arctan2(_v[1, _i], _v[0, _i]))
                e_samples.append(np.sqrt(1. - _w[_j]**2 / _w[_i]**2))

            # to get proper error for theta, split into x- and y-components, calculate
            # uncertainty for each component, and perform error propagation
            mean_x = np.cos(theta + np.pi / 2.)
            mean_y = np.sin(theta + np.pi / 2.)
            scatter_x = np.cos(theta_samples).std()
            scatter_y = np.sin(theta_samples).std()
            theta_err = np.sqrt(mean_y ** 2 * scatter_x ** 2 / mean_x ** 4 + scatter_y ** 2 / mean_x ** 2) / (
                    1. + mean_y ** 2 / mean_x ** 2)

            e_err = np.std(e_samples)

            results[ii][['e_err', 'theta_err']] = [e_err, theta_err*u.rad]

        results['frac'][ii] = min(1., hull.volume / (np.pi*r_max.value**2))

    return results
