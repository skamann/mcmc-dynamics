import logging
from astropy import units as u


logger = logging.getLogger(__name__)


def get_perspective_rotation(dx, dy, mu_x, mu_y, d):
    """
    Determine apparent rotation along the line of sight caused by the proper
    motion of the cluster.

    See equation 6 in van de Ven et al. (2006).

    Parameters
    ----------
    dx : astropy.unit.Quantity
        The offset from the cluster centre in x-direction (i.e. in -RA
        direction).
    dy : astropy.unit.Quantity
        The offset from the cluster centre in y-direction (i.e. in +Dec
        direction).
    mu_x : astropy.unit.Quantity
        The proper motion of the cluster in x-direction (i.e. in -RA
        direction).
    mu_y : astropy.unit.Quantity
        The proper motion of the cluster in y-direction (i.e. in +Dec
        direction).
    d: astropy.unit.Quantity
        The distance to the cluster.

    Returns
    -------
    v_los : astropy.unit.Quantity
        The apparent velocity at each provided offset caused by the proper
        motion of the cluster.
    """
    dx = u.Quantity(dx)
    if dx.unit.is_unity():
        dx *= u.arcmin
        logger.warning('No unit provided for parameter <dx>. Assuming {0}.'.format(dx.unit))
    dy = u.Quantity(dy)
    if dy.unit.is_unity():
        dy *= u.arcmin
        logger.warning('No unit provided for parameter <dy>. Assuming {0}.'.format(dy.unit))

    mu_x = u.Quantity(mu_x)
    if mu_x.unit.is_unity():
        mu_x *= u.mas/u.yr
        logger.warning('No unit provided for parameter <mu_x>. Assuming {0}.'.format(mu_x.unit))
    mu_y = u.Quantity(mu_y)
    if mu_y.unit.is_unity():
        mu_y *= u.mas / u.yr
        logger.warning('No unit provided for parameter <mu_y>. Assuming {0}.'.format(mu_y.unit))

    d = u.Quantity(d)
    if d.unit.is_unity():
        d *= u.kpc
        logger.warning('No unit provided for parameter <d>. Assuming {0}.'.format(d.unit))

    return 1.3790e-3 * u.km/u.s * d.to(u.kpc).value * (
            dx.to(u.arcmin).value * mu_x.to(u.mas/u.yr).value + dy.to(u.arcmin).value * mu_y.to(u.mas/u.yr).value)
