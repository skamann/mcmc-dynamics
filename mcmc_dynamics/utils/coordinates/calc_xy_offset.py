import logging
import numpy as np
from astropy import units as u


logger = logging.getLogger(__name__)


def calc_xy_offset(ra, dec, ra_center, dec_center):
    """Returns the (x,y) offset to the cluster centre in arcmin from (RA, Dec), see van de Ven+ (2006)"""
    r0 = 10800.*u.arcmin / np.pi

    ra = u.Quantity(ra)
    if ra.unit.is_unity():
        ra *= u.deg
        logger.warning('No unit provided for parameter <ra>. Assuming {0}.'.format(ra.unit))
    dec = u.Quantity(dec)
    if dec.unit.is_unity():
        dec *= u.deg
        logger.warning('No unit provided for parameter <dec>. Assuming {0}.'.format(dec.unit))
    ra_center = u.Quantity(ra_center)
    if ra_center.unit.is_unity():
        ra_center *= u.deg
        logger.warning('No unit provided for parameter <ra_center>. Assuming {0}.'.format(ra_center.unit))
    dec_center = u.Quantity(dec_center)
    if dec_center.unit.is_unity():
        dec_center *= u.deg
        logger.warning('No unit provided for parameter <dec_center>. Assuming {0}.'.format(dec_center.unit))

    dx = -r0 * np.cos(dec) * np.sin(ra - ra_center)
    dy = r0 * (np.sin(dec) * np.cos(dec_center) - np.cos(dec) * np.sin(dec_center) * np.cos(ra - ra_center))

    return dx, dy
