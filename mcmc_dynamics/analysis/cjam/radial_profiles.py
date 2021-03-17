import importlib.resources as pkg_resources
import numpy as np
from scipy import stats
from astropy import units as u

from .axisymmetric import Axisymmetric
from ... import config
from ...utils.files import get_nearest_neigbhbour_idx2, get_mge
from ...parameter import Parameters


class RadialProfiles(Axisymmetric):
    """
    This class implements axisymmetric Jeans models with radially varying
    mass-to-light ratio and rotation parameter kappa.

    The radial variation is achieved by assigning different mass-to-light
    ratios and kappa parameters to different components of the MGE profile.
    The values of MGE components for which either the mass-to-light ratio
    and/or kappa are obtained via linear interpolation.

    In any case, the mass-to-light ratio (`mlr_1`) and the kappa parameter
    (`kappa_1`) of the first MGE component must be specified. For all other
    components, specifying `mlr_{n}` and `kappa_{n}` is optional.
    """
    MODEL_PARAMETERS = ['d', 'mlr_1', 'barq', 'kappa_x', 'kappa_y', 'beta', 'mbh',
                        'delta_x', 'delta_y', 'rbh', 'delta_v']

    def __init__(self, data, mge_mass, mge_lum, parameters=None, **kwargs):

        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'radial_profiles.json'))

        super(RadialProfiles, self).__init__(data=data, mge_mass=mge_mass, mge_lum=mge_lum, parameters=parameters,
                                             **kwargs)

    # @property
    # def parameter_labels(self):
    #     labels = {}
    #     for row in self.initials:
    #         name = row['name']
    #         latex_string = row['init'].unit.to_string('latex')
    #         if name == 'd':
    #             labels[name] = r'$d/${0}'.format(latex_string)
    #         elif len(name) > 3 and name[:3] == 'mlr':
    #             index = int(name[3:])
    #             labels[name] = r'$\Upsilon_{0}/\frac{{\rm M_\odot}}{{\rm L_\odot}}$'.format(index)
    #         elif row['name'] == 'barq':
    #             labels[row['name']] = r'$\bar{q}$'
    #         elif len(name) > 5 and name[:5] == 'kappa':
    #             index = int(name[5:])
    #             labels[row['name']] = r'$\kappa_{0}$'.format(index)
    #         elif row['name'] == 'beta':
    #             labels[row['name']] = r'$\beta$'
    #         elif row['name'] == 'mbh':
    #             labels[row['name']] = r'$M_{\rm BH}$'
    #         elif row['name'] == 'delta_x':
    #             labels[row['name']] = r'$\Delta x$'
    #         elif row['name'] == 'delta_y':
    #             labels[row['name']] = r'$\Delta y$'
    #
    #         else:
    #             labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
    #     return labels

    def fetch_parameter_values(self, values):

        parameters = super(RadialProfiles, self).fetch_parameter_values(values)

        # collect all kappa and mlr values in arrays
        mlr = np.zeros((self.mge_mass.n_components, ), dtype=np.float64)*self.parameters['mlr_1']
        kappa = np.zeros((self.mge_lum.n_components, ), dtype=np.float64)*self.parameters['kappa_1']

        defined_mlr = np.zeros((self.mge_mass.n_components, ), dtype=np.bool)
        defined_kappa = np.zeros((self.mge_lum.n_components, ), dtype=np.bool)

        for i in range(self.mge_mass.n_components):
            if 'mlr_{0}'.format(i + 1) in parameters.keys():
                mlr[i] = parameters.pop('mlr_{0}'.format(i + 1))
                defined_mlr[i] = True
        for i in range(self.mge_lum.n_components):
            if 'kappa_{0}'.format(i + 1) in parameters.keys():
                kappa[i] = parameters.pop('kappa_{0}'.format(i + 1))
                defined_kappa[i] = True

        # interpolate missing values linearly in log(r) space
        mlr[~defined_mlr] = np.interp(np.log10(self.mge_mass.data['s'][~defined_mlr].value),
                                      np.log10(self.mge_mass.data['s'][defined_mlr].value),
                                      mlr[defined_mlr])*mlr.unit
        kappa[~defined_kappa] = np.interp(np.log10(self.mge_lum.data['s'][~defined_kappa].value),
                                          np.log10(self.mge_lum.data['s'][defined_kappa].value),
                                          kappa[defined_kappa])*kappa.unit

        parameters['mlr'] = mlr
        parameters['kappa'] = kappa

        return parameters

    def lnprior(self, values):
        """
        Evaluate prior, make sure that no MGE component has negative
        mass-to-light ratio.

        Parameters
        ----------
        values : array_like
            The set of parameter values for which to evaluate the prior.

        Returns
        -------
        lnprior : float
           The prior of the model for the given set of parameter values.
        """
        mlr_values = self.fetch_parameter_values(values)['mlr']
        if (mlr_values <= 0.1).any():
            return -np.inf
        return super(RadialProfiles, self).lnprior(values=values)


class AnalyticalProfiles(Axisymmetric):
    """
    This class implements axisymmetric Jeans models with radially varying
    mass-to-light ratio and rotation parameter kappa.

    The radial variation is achieved by defining analytical functions for the
    mass-to-light ratio and the kappa-parameter.

    For the mass-to-light ratio, the following formula is used:

    mlr(r) = (mlr_0*(1.-R) + 2.*mlr_t*R + mlr_inf*R*(R-1.)) / (1.+R**2),

    with R = r/r_mlr. The parameters `r_mlr`, `mlr_0`, `mlr_t`, and `mlr_inf`
    are the model parameters that can be optimized during the analysis.

    For kappa, the following formula is used:

    kappa(r) = 2.* kappa_max * (r/r_kappa) / (1. + (r/r_kappa)**2).

    In order to be able to fit the position angle theta of the rotation,
    kappa_max is parametrized as kappa_max = SQRT(kappa_x**2 + kappa_y**2),
    with theta = ARCTAN(kappa_x/kappa_x). The parameters `kappa_x`, `kappa_y`,
    and `r_kappa` are the model parameters that can be optimized during the
    analysis.
    """
    MODEL_PARAMETERS = ['d', 'mlr_0', 'mlr_t', 'mlr_inf', 'r_mlr', 'barq', 'kappa_x', 'kappa_y', 'r_kappa',
                        'beta', 'mbh', 'delta_x', 'delta_y', 'rbh', 'delta_v']

    def __init__(self, data, mge_mass, mge_lum, parameters=None, mge_files=None, **kwargs):
        """
        Initialize a new instance of the AnalyticalProfiles class.

        Parameters
        ----------
        data
        mge_mass
        mge_lum
        parameters
        mge_files
        kwargs
        """
        if parameters is None:
            parameters = Parameters().load(pkg_resources.open_text(config, 'analytical_profiles.json'))

        super(AnalyticalProfiles, self).__init__(data=data, mge_mass=mge_mass, mge_lum=mge_lum,
                                                 mge_files=mge_files, parameters=parameters, **kwargs)
        
    @staticmethod
    def calculate_x_values(single_mge):
        """
        For a given MGE, the code determines the radii at which each MGE
        component contributes maximally to the combined profile.

        Parameters
        ----------
        single_mge : instance of MgeReader
            The MGE profile for which the radii should be determined.

        Returns
        -------
        xn : array_like
            The radii where the contribution of each MGE component to the full
            profile is maximized.
        """
        x = np.logspace(u.Dex(single_mge.data['s']).min().value,
                        u.Dex(single_mge.data['s']).max().value,
                        100)*single_mge.data['s'].unit
        
        weights = np.zeros((x.size, single_mge.n_components))
        for i in range(single_mge.n_components):
            weights[:, i] = single_mge.data['i'][i] * np.exp(-0.5 * (x / single_mge.data['s'][i]) ** 2)
        weights /= weights.sum(axis=1)[:, np.newaxis]
        
        xn = x[weights.argmax(axis=0)]
        xn[single_mge.data['s'].argmin()] = 0
        xn[single_mge.data['s'].argmax()] *= 10
        
        return xn

    # @property
    # def parameter_labels(self):
    #     labels = {}
    #     for row in self.initials:
    #         name = row['name']
    #         latex_string = row['init'].unit.to_string('latex')
    #         if name == 'd':
    #             labels[name] = r'$d/${0}'.format(latex_string)
    #         elif len(name) > 3 and name[:3] == 'mlr':
    #             suffix = name[4:]
    #             if suffix == 'inf':
    #                 suffix = r'\infty'
    #             labels[name] = r'$\Upsilon_{{\rm {0}}}/{{\rm M_\odot}}\,{{\rm L_\odot^{{-1}}}}$'.format(suffix)
    #         elif name == 'barq':
    #             labels[name] = r'$\bar{q}$'
    #         elif name == 'kappa_x':
    #             labels[name] = r'$\kappa_{\rm x}$'
    #         elif name == 'kappa_y':
    #             labels[name] = r'$\kappa_{\rm y}$'
    #         elif name == 'beta':
    #             labels[name] = r'$\beta$'
    #         elif name == 'r_kappa':
    #             labels[name] = r'$r_{{\rm \kappa}}$/{0}'.format(latex_string)
    #         elif name == 'r_mlr':
    #             labels[name] = r'$r_{{\rm \Upsilon}}$/{0}'.format(latex_string)
    #         elif row['name'] == 'delta_x':
    #             labels[row['name']] = r'$\Delta x$'
    #         elif row['name'] == 'delta_y':
    #             labels[row['name']] = r'$\Delta y$'
    #         elif row['name'] == 'mbh':
    #             labels[row['name']] = r'$M_{\rm BH}$'
    #         else:
    #             labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
    #     return labels

    def fetch_parameter_values(self, values, return_rkappa=False, return_mge=False):

        parameters = super(AnalyticalProfiles, self).fetch_parameter_values(values)
        
        if self.use_mge_grid:
            # find out which MGE profile to use based on the current offset
            idx = get_nearest_neigbhbour_idx2(-parameters['delta_x'].to(u.arcsec).value, 
                                              -parameters['delta_y'].to(u.arcsec).value, 
                                              self.mge_files)
            mge_lum, mge_mass = get_mge(self.mge_files[idx])
        else:
            mge_lum, mge_mass = self.mge_lum, self.mge_mass

        # MGE components are assigned the values of the analytical functions at the distances where their contribution
        # to the overall profiles is max.            
        _x_mlr = AnalyticalProfiles.calculate_x_values(mge_lum)
        _x_kappa = AnalyticalProfiles.calculate_x_values(mge_mass)

        _x = _x_mlr/parameters.pop('r_mlr')
        parameters['mlr'] = (
            parameters.pop('mlr_0')*(1.-_x) + 2.*parameters.pop('mlr_t')*_x + parameters.pop('mlr_inf')*_x*(_x-1.))/(
                1.+_x**2)

        rkappa = parameters.pop('r_kappa')
        _x = (_x_kappa / rkappa).si
        kappa_max = np.sqrt(parameters['kappa_x']**2 + parameters['kappa_y']**2)
        parameters['kappa'] = 2. * kappa_max * _x / (1. + _x**2)
        
        if return_rkappa:
            return parameters, rkappa
        
        if return_mge:
            return parameters, mge_lum, mge_mass
        
        return parameters

    # def lnprior(self, values):
    #     # some priors depend on the current MGE profile, so we need to get it
    #     # (since it is not saved as self.xyz)
    #     if self.use_mge_grid:
    #         _, mge_lum, mge_mass = self.fetch_parameter_values(values, return_mge=True)
    #     else:
    #         mge_lum, mge_mass = self.mge_lum, self.mge_mass
    #
    #     # add additional checks for parameters of radial profiles
    #     r_mlr = u.Quantity(self.parameters['r_mlr'].value, unit=self.parameters['r_mlr'].unit)
    #     if not (mge_mass.data['s'].min() < r_mlr < mge_mass.data['s'].max()):
    #         print('r_mlr', r_mlr, mge_mass.data['s'].min(), mge_mass.data['s'].max())
    #         return -np.inf
    #
    #     r_kappa = u.Quantity(self.parameters['r_kappa'].value, unit=self.parameters['r_kappa'].unit)
    #     if not (mge_lum.data['s'].min() < r_kappa < mge_lum.data['s'].max()):
    #         print('r_kappa', r_kappa, mge_lum.data['s'].min(), mge_lum.data['s'].max())
    #         return -np.inf
    #
    #     return super(AnalyticalProfiles, self).lnprior(values=values)
