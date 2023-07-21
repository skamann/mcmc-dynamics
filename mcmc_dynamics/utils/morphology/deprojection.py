import logging
import numpy as np


logger = logging.getLogger(__name__)


def find_barq_limits(q, parameters=None):
    """
    For de-projecting the MGE, q = SQRT(q'^2 - COS(incl)^2)/SIN(incl) needs to
    be evaluated for each component. As an absolute lower limit, the
    inclination therefore cannot be < ARCCOS(q'_min). However, the JAM call
    fails if any of the de-projected MGE components have q<0.05. This imposes
    a stricter limit, namely COS(incl)^2 > (q'_min^2 - 0.05^2)/(1. - 0.05^2).
    """
    median_q = np.median(q)
    min_q = np.min(q)

    lower_limit_q_deprojected = 0.05

    if min_q < 1:
        min_cosi2 = (min_q**2 - lower_limit_q_deprojected ** 2) / (1. - lower_limit_q_deprojected ** 2)
        barq_min = np.sqrt((median_q ** 2 - min_cosi2) / (1. - min_cosi2))
    else:
        barq_min = 0
    barq_max = median_q

    # make sure limits on barq are correct in the Parameters instance if provided.
    if parameters is not None:
        if parameters['barq'].max > barq_max:
            logger.warning(f"Setting upper limit for parameter 'barq' to {barq_max:.3f}.")
            parameters['barq'].set(max=barq_max)
        if parameters['barq'].min < barq_min:
            logger.warning(f"Setting lower limit for parameter 'barq' to {barq_min:.3f}.")
            parameters['barq'].set(min=barq_min)

    return barq_min, barq_max
