# -*- coding: utf-8 -*-
import inspect
import logging
import numpy as np
try:
    from importlib.resources import files
except ImportError:  # for Python v<3.9
    from importlib_resources import files
from astropy import units as u
from astropy.table import Table
from .constant import ConstantFit
from .. import config
from ..parameter import Parameters
from ..utils.files import DataReader
from ..utils.coordinates import calc_xy_offset


logger = logging.getLogger(__name__)


class ModelFit(ConstantFit):
    """
    The purpose of the ModelFit class is to fit the radial rotation and
    dispersion profile of a stellar population using simple analytical models.

    The rotation profile is modeled as expected for a system that underwent
    violent relaxation (e.g., [Lynden-Bell (1967)](#references)). In this case,
    the radial dependence is given as

    $$
    v_{rot}(r, \\theta) = v_{\\rm SYS} + 2(v_{\\rm MAX}/r_{\\rm PEAK}) \\cdot
    x_{\\rm PA}/(1 + (x_{\\rm PA}/r_{\\rm PEAK})^2),
    $$

    where

    $$
    x_{\\rm PA}(r, \\theta) = r \\cdot \\sin(\\theta - \\theta_0).
    $$

    The dispersion is modeled as a [Plummer (1911)](#references) profile with
    the following functional form,

    $$
    \\sigma(r) = \\sigma_0/(1 + r^2 / a^2)^{0.25}.
    $$

    As fitting angles can be difficult due to the discontinuity at $2\\pi$,
    instead of $\\theta_0$, the $x$ and $y$ components of $v_{\\rm MAX}$ are
    considered free parameters. Hence, the model has the following 8
    parameters that can be optimized. `v_sys`, `v_maxx`, `v_maxy`, `r_peak`,
    `sigma_max`, `a`, `ra_center`, `dec_center`.

    The data required per star are the world coordinates $\\alpha$ and
    $\\delta$, given in decimal degrees, the radial velocities $v$ and the
    velocity uncertainty $\\epsilon_{\\rm v}$.

    ### References

    - [Lynden-Bell (1967)](https://ui.adsabs.harvard.edu/abs/1967MNRAS.136..101L/abstract)
    - [Plummer (1911)](https://ui.adsabs.harvard.edu/abs/1911MNRAS..71..460P/abstract)

    """
    MODEL_PARAMETERS = ['v_sys', 'v_maxx', 'v_maxy', 'r_peak', 'sigma_max', 'a', 'ra_center', 'dec_center']
    OBSERVABLES = {'v': u.km/u.s, 'verr': u.km/u.s, 'ra': u.deg, 'dec': u.deg}

    parameters_file = files(config).joinpath('model.json')

    def __init__(self, data: DataReader, parameters: Parameters = None, **kwargs):
        """
        :param data: The observed data for a set of n stars. The instance must
            provide at least the coordinates, the velocities, and their
            uncertainties.
        :param parameters: The model parameters.
        :param kwargs: Any additional keyword arguments are passed on to the
            initialization of the parent class.
        """
        if parameters is None:
            parameters = Parameters().load(self.parameters_file)

        super(ModelFit, self).__init__(data=data, parameters=parameters, **kwargs)

        # get parameters required to evaluate rotation and dispersion models
        self.rotation_parameters = inspect.signature(self.rotation_model).parameters
        self.dispersion_parameters = inspect.signature(self.dispersion_model).parameters

    def dispersion_model(self, sigma_max: float, ra_center: float, dec_center: float, a: float = 1.,
                         **kwargs) -> np.ndarray:
        """
        Calculate the line-of-sight velocity dispersion at the positions of
        the available data points.

        :param sigma_max: The central velocity dispersion of the model.
        :param ra_center: The right ascension of the model centre.
        :param dec_center: The declination of the model centre.
        :param a: The scale radius of the model.
        :param kwargs: This method does not use any additional keyword
            arguments.

        :return: The line-of-sight velocity dispersion of the model evaluated
            at the positions of the individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.dispersion_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        dx, dy = calc_xy_offset(ra=self.ra, dec=self.dec, ra_center=ra_center, dec_center=dec_center)
        r = np.sqrt(dx**2 + dy**2)
        return sigma_max / (1. + r ** 2 / a ** 2) ** 0.25

    def rotation_model(self, v_sys: float, v_maxx: float, v_maxy: float, ra_center: float, dec_center: float,
                       r_peak: float = None, **kwargs) -> np.ndarray:
        """
        Calculate the line-of-sight velocity at the positions of the available
        data points.

        :param v_sys: The constant systemic velocity of the model.
        :param v_maxx: The x-component of the rotation amplitude of the model.
        :param v_maxy: The y-component of the rotation amplitude of the model.
        :param ra_center: The right ascension of the model centre.
        :param dec_center: The declination of the model centre.
        :param r_peak: The position of the peak of the rotation curve.
        :param kwargs: This model does not use any additional keyword arguments.

        :return: The model velocity along the line-of-sight at each of the
            individual data points.
        """
        if kwargs:
            raise IOError('Unknown keyword argument(s) "{0}" for method {1}.rotation_model.'.format(
                ', '.join(kwargs.keys()), self.__class__.__name__))

        dx, dy = calc_xy_offset(ra=self.ra, dec=self.dec, ra_center=ra_center, dec_center=dec_center)
        r = np.sqrt(dx**2 + dy**2)
        if r_peak is None:
            r_peak = np.median(r)

        v_max = np.sqrt(v_maxx**2 + v_maxy**2)
        theta_0 = np.arctan2(v_maxy, v_maxx)
        theta = np.arctan2(dy, dx)
        x_pa = r * np.sin(theta - theta_0)
        return v_sys + 2. * (v_max / r_peak) * x_pa / (1. + (r / r_peak) ** 2)

    def create_profiles(self, chain: np.ndarray, n_burn: int, radii: np.ndarray = None,
                        filename: str = None) -> Table:
        """
        Convert the parameter distributions returned by the MCMC analysis
        into radial profiles of the rotation amplitude and velocity
        dispersion.

        :param chain: The chains produced by the MCMC sampler. They should be
            provided as a 3D array, containing the parameters as first index,
            the steps as second index, and the chains as third index.
        :param n_burn: The number of steps that are ignored at the beginning
            of each MCMC chain.
        :param radii: The radii at which the profiles should be calculated.
        :param filename: Name of a csv-file in which the resulting profiles
            will be stored.

        :return: The output table will contain the values of the
            rotation amplitude and the dispersion predicted by the model for
            the requested radii. It will further contain the lower and upper
            1-sigma and 3-sigma limits for those values.
        """
        # collect parameters
        fitted_models = {}

        i = 0
        '''
        do_later = []
        params = self.parameters.copy()
        for name, parameter in params.items():
            if parameter.fixed:
                if parameter.expr is not None:
                    do_later.append(parameter)
            else:
                parameter.value = u.Quantity(chains[:, n_burn:, i].flatten(), parameter.unit)
                i += 1

        for parameter in do_later:
            parameter.__getstate__()

        for name, parameter in params.items():
            fitted_models[name] = u.Quantity(parameter.value, parameter.unit)
        '''
        for name, parameter in self.parameters.items():
            if parameter.fixed:
                fitted_models[name] = u.Quantity(parameter.value, parameter.unit)
            else:
                fitted_models[name] = u.Quantity(chain[:, n_burn:, i].flatten(), parameter.unit)
                i += 1

        v_maxx = fitted_models['v_maxx']
        v_maxy = fitted_models['v_maxy']
        r_peak = fitted_models['r_peak']
        sigma_max = fitted_models['sigma_max']
        a = fitted_models['a']

        if radii is None:
            radii = np.logspace(-1, 2.5, 50)*u.arcsec
        else:
            radii = u.Quantity(radii)
            if radii.unit is u.dimensionless_unscaled:
                radii = radii*r_peak.unit

        v_max = np.sqrt(v_maxx**2 + v_maxy**2)
        v_rot = 2. * (v_max / r_peak) * radii[:, np.newaxis] / (1. + (radii[:, np.newaxis] / r_peak) ** 2)
        pv_rot = np.percentile(v_rot.to(u.km/u.s), [50, 16, 84, 0.15, 99.85], axis=-1)
        sigma = sigma_max / (1. + radii[:, np.newaxis] ** 2 / a ** 2) ** 0.25
        psigma = np.percentile(sigma.to(u.km/u.s), [50, 16, 84, 0.15, 99.85], axis=-1)

        profile = Table([
            Table.Column(radii, name='r'),
            Table.Column(pv_rot[0], name='v_rot', unit=u.km/u.s),
            Table.Column(pv_rot[1], name='v_rot_lower_1s', unit=u.km / u.s),
            Table.Column(pv_rot[2], name='v_rot_upper_1s', unit=u.km / u.s),
            Table.Column(pv_rot[3], name='v_rot_lower_3s', unit=u.km / u.s),
            Table.Column(pv_rot[4], name='v_rot_upper_3s', unit=u.km / u.s),
            Table.Column(psigma[0], name='sigma', unit=u.km / u.s),
            Table.Column(psigma[1], name='sigma_lower_1s', unit=u.km / u.s),
            Table.Column(psigma[2], name='sigma_upper_1s', unit=u.km / u.s),
            Table.Column(psigma[3], name='sigma_lower_3s', unit=u.km / u.s),
            Table.Column(psigma[4], name='sigma_upper_3s', unit=u.km / u.s),
        ])

        if filename is not None:
            profile.write(filename, format='ascii.ecsv', overwrite=True)

        return profile
