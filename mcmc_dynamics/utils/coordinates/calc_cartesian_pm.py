import logging
import numpy as np
from astropy import units as u


logger = logging.getLogger(__name__)


def calc_cartesian_pm(pmra, pmdec, ra, dec, ra0, dec0):
    """
    Convert proper motions measured along RA and Dec to proper motions along
    the x- and y-axes of a cartesian coordinate system.

    Parameters
    ----------
    pmra : astropy.unit.Quantity
        The proper motion in RA-direction (pmra in the Gaia catalog).
    pmdec : astropy.unit.Quantity
        The proper motion in Dec-direction (pmdec in the Gaia catalog).
    ra : astropy.unit.Quantity
        The right ascensions where the proper motions have been measured.
    dec : astropy.unit.Quantity
        The declinations were the proper motions have been measured.
    ra0 : astropy.unit.Quantity
        The RA-coordinate where the origin of the cartesian coordinate system
        is located.
    dec0 : astropy.unit.Quantity
        The Dec-coordinate where the origin of the cartesian coordinate system
        is located.

    Returns
    -------
    pmx : astropy.unit.Quantity
        The proper motion along the y-axis of the cartesian coordinate system.
    pmy : astropy.unit.Quantity
        The proper motion along the y-axis of the cartesian coordinate system.
    """
    pmra = u.Quantity(pmra)
    if pmra.unit.is_unity():
        pmra *= u.mas/u.yr
        logger.warning('Missing unit for parameter <pmra>. Assuming {0}.'.format(pmra.unit))
    pmdec = u.Quantity(pmdec)
    if pmdec.unit.is_unity():
        pmdec *= u.mas/u.yr
        logger.warning('Missing unit for parameter <pmdec>. Assuming {0}.'.format(pmdec.unit))

    ra = u.Quantity(ra)
    if ra.unit.is_unity():
        ra *= u.deg
        logger.warning('Missing unit for parameter <ra>. Assuming {0}.'.format(ra.unit))
    dec = u.Quantity(dec)
    if dec.unit.is_unity():
        dec *= u.deg
        logger.warning('Missing unit for parameter <dec>. Assuming {0}.'.format(dec.unit))

    ra0 = u.Quantity(ra0)
    if ra0.unit.is_unity():
        ra0 *= u.deg
        logger.warning('Missing unit for parameter <ra0>. Assuming {0}.'.format(ra0.unit))
    dec0 = u.Quantity(dec0)
    if dec0.unit.is_unity():
        dec0 *= u.deg
        logger.warning('Missing unit for parameter <dec0>. Assuming {0}.'.format(dec0.unit))

    # following equations taken from the Gaia DR2 paper (Helmi et al. 2018, equation 2). However the sign of
    # pmra has been switched because the x-axis in our coordinate system increases from east to west
    pmx = -pmra * np.cos(ra - ra0) - pmdec * np.sin(dec) * np.sin(ra - ra0)
    pmy = -pmra * np.sin(dec0) * np.sin(ra - ra0) + pmdec * (
            np.cos(dec) * np.cos(dec0) + np.sin(dec) * np.sin(dec0) * np.cos(ra - ra0))
    return pmx, pmy

    # ToDo: error propagation
    # if pmra_err is None or pmdec_err is None:
    #     return pmx, pmy, None, None
    # else:
    #     pmx_err = np.sqrt((pmra_err * np.cos(_ra - _ra0)) ** 2 + (pmdec_err * np.sin(_dec) * np.sin(_ra - _ra0)) ** 2)
    #     pmy_err = np.sqrt((pmra_err * np.sin(_dec0) * np.sin(_ra - _ra0)) ** 2 + (pmdec_err * (
    #             np.cos(_dec) * np.cos(_dec0) + np.sin(_dec) * np.sin(_dec0) * np.cos(_ra - _ra0))) ** 2)
    #     return pmx, pmy, pmx_err, pmy_err