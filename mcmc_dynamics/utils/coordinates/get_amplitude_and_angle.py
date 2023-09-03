import logging
import numpy as np
from astropy import units as u
from astropy.table import QTable


logger = logging.getLogger(__name__)


def get_amplitude_and_angle(pars, return_samples=False):

    if 'theta_0' not in pars and 'v_maxx' in pars and 'v_maxy' in pars:
        pars['theta_0'] = np.arctan2(pars['v_maxy'], pars['v_maxx'])
    elif 'v_maxx' not in pars and 'theta_0' in pars and 'v_maxy' in pars:
        pars['v_maxx'] = pars['v_maxy'] * np.tan(pars['theta_0'])
    elif 'v_maxy' not in pars and 'theta_0' in pars and 'v_maxx' in pars:
        pars['v_maxy'] = pars['v_maxx'] / np.tan(pars['theta_0'])

    for par in ['theta_0', 'v_maxx', 'v_maxy']:
        if par not in pars:
            logger.error('Failed to recover parameter {}.'.format(par))
            return None, None, None

    # make sure median angle is in the centre of the full angle range (i.e. at 0 when range is (-Pi, Pi])
    median_theta = np.arctan2(np.median(pars['v_maxy']), np.median(pars['v_maxx']))
    _theta = pars['theta_0'] - median_theta
    _theta = np.where(_theta < -np.pi, _theta + 2 * np.pi, _theta)
    _theta = np.where(_theta > np.pi, _theta - 2 * np.pi, _theta)

    # to obtain v_max, the values of (v_maxx, vmaxy) rotated by -median_theta. That way, one component will be
    # in direction of median_theta, which we consider as v_max.
    v_max = pars['v_maxx'] * np.cos(-median_theta) - pars['v_maxy'] * np.sin(-median_theta)

    results = QTable(data=[['median', 'uperr', 'loerr']], names=['value'])
    results.add_index('value')

    for name, values in {'v_max': v_max, 'theta_0': _theta}.items():
        unit = u.rad if name == 'theta_0' else u.dimensionless_unscaled

        percentiles = np.percentile(values, [16, 50, 84])

        results.add_column(QTable.Column(
            [percentiles[1], percentiles[2] - percentiles[1], percentiles[1] - percentiles[0]] * unit,
            name=name))

    results.loc['median']['theta_0'] += median_theta * u.rad

    if return_samples:
        return results, v_max, _theta
    else:
        return results, None, None
