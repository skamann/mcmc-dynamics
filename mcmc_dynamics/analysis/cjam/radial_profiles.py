import numpy as np
from scipy import stats
from astropy import units as u
from .axisymmetric import Axisymmetric
from mcmc_dynamics.utils.files import get_nearest_neigbhbour_idx


class RadialProfiles(Axisymmetric):

    def __init__(self, data, mge_mass, mge_lum, initials, **kwargs):

        super(RadialProfiles, self).__init__(data=data, mge_mass=mge_mass, mge_lum=mge_lum, initials=initials, **kwargs)

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(Axisymmetric, self).parameters
            _ = self._parameters.pop('mlr')
            _ = self._parameters.pop('kappa')
            self._parameters.update({'mlr1': u.dimensionless_unscaled, 'mlr3': u.dimensionless_unscaled,
                                     'mlr4': u.dimensionless_unscaled, 'mlr7': u.dimensionless_unscaled,
                                     'kappa3': u.dimensionless_unscaled})
        return self._parameters

    @property
    def parameter_labels(self):
        labels = {}
        for row in self.initials:
            name = row['name']
            latex_string = row['init'].unit.to_string('latex')
            if name == 'd':
                labels[name] = r'$d/${0}'.format(latex_string)
            elif len(name) > 3 and name[:3] == 'mlr':
                index = int(name[3:])
                labels[name] = r'$\Upsilon_{0}/\frac{{\rm M_\odot}}{{\rm L_\odot}}$'.format(index)
            elif row['name'] == 'barq':
                labels[row['name']] = r'$\bar{q}$'
            elif len(name) > 5 and name[:5] == 'kappa':
                index = int(name[5:])
                labels[row['name']] = r'$\kappa_{0}$'.format(index)
            elif row['name'] == 'beta':
                labels[row['name']] = r'$\beta$'
            elif row['name'] == 'mbh':
                labels[row['name']] = r'$M_{\rm BH}$'
            elif row['name'] == 'delta_x':
                labels[row['name']] = r'$\Delta x$'
            elif row['name'] == 'delta_y':
                labels[row['name']] = r'$\Delta y$'

            else:
                labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
        return labels

    def fetch_parameters(self, values):

        parameters = super(RadialProfiles, self).fetch_parameters(values)

        # collect all kappa and mlr values in arrays
        mlr = np.zeros((self.mge_mass.n_components, ), dtype=np.float64)*self.parameters['mlr1']
        kappa = np.zeros((self.mge_lum.n_components, ), dtype=np.float64)*self.parameters['kappa3']

        defined_mlr = np.zeros((self.mge_mass.n_components, ), dtype=np.bool)
        defined_kappa = np.zeros((self.mge_lum.n_components, ), dtype=np.bool)

        for i in range(self.mge_lum.n_components):
            if 'mlr{0}'.format(i + 1) in parameters.keys():
                mlr[i] = parameters.pop('mlr{0}'.format(i + 1))
                defined_mlr[i] = True
            if 'kappa{0}'.format(i + 1) in parameters.keys():
                kappa[i] = parameters.pop('kappa{0}'.format(i + 1))
                defined_kappa[i] = True

        # interpolate missing values
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

        for parameter, value in self.fetch_parameters(values).items():
            if parameter == 'd' and value <= 0.5*u.kpc:
                return -np.inf
            elif parameter == 'mlr' and (value <= 0.1).any():
                return -np.inf
            elif parameter == 'barq' and (value <= 0.2 or value > self.median_q):
                return -np.inf
            if parameter == 'kappa' and (abs(value) > 10).any():
                return -np.inf
        return super(RadialProfiles, self).lnprior(values=values)


class AnalyticalProfiles(Axisymmetric):

    def __init__(self, data, mge_mass, mge_lum, initials, mge_coords=None, **kwargs):

        super(AnalyticalProfiles, self).__init__(data=data, mge_mass=mge_mass, mge_lum=mge_lum, 
                                                 mge_coords=mge_coords, initials=initials,
                                                 **kwargs)
        
        self.mge = dict()
        if self.use_mge_grid:
            idx = get_nearest_neigbhbour_idx(0, 0, self.mge_coords)
            self.mge['mass_s'] = self.mge_mass[idx].data['s']
            self.mge['mass_i'] = self.mge_mass[idx].data['i']
            self.mge['mass_n_components'] = self.mge_mass[idx].n_components
            
            self.mge['lum_s'] = self.mge_lum[idx].data['s']
            self.mge['lum_i'] = self.mge_lum[idx].data['i']
            self.mge['lum_n_components'] = self.mge_lum[idx].n_components
        
        else:
            self.mge['mass_s'] = self.mge_mass.data['s']
            self.mge['mass_i'] = self.mge_mass.data['i'] 
            self.mge['mass_n_components'] = self.mge_mass.n_components
            
            self.mge['lum_s'] = self.mge_lum.data['s']
            self.mge['lum_i'] = self.mge_lum.data['i']
            self.mge['lum_n_components'] = self.mge_lum.n_components

        # MGE components are assigned the values of the analytical functions at the distances where their contribution
        # to the overall profiles is max.
        x = np.logspace(u.Dex(self.mge['mass_s']).min().value,
                        u.Dex(self.mge['mass_s']).max().value,
                        100)*self.mge['mass_s'].unit
        
        weights = np.zeros((x.size, self.mge['mass_n_components']))
        for i in range(self.mge['mass_n_components']):
            weights[:, i] = self.mge['mass_i'][i] * np.exp(-0.5 * (x / self.mge['mass_s'][i]) ** 2)
        weights /= weights.sum(axis=1)[:, np.newaxis]
        
        self.x_mlr = x[weights.argmax(axis=0)]
        self.x_mlr[self.mge['mass_s'].argmin()] = 0.
        self.x_mlr[self.mge['mass_s'].argmax()] *= 10

        x = np.logspace(u.Dex(self.mge['lum_s']).min().value,
                        u.Dex(self.mge['lum_s']).max().value,
                        100)*self.mge['lum_s'].unit
        
        weights = np.zeros((x.size, self.mge['lum_n_components']))
        for i in range(self.mge['lum_n_components']):
            weights[:, i] = self.mge['lum_i'][i] * np.exp(-0.5 * (x / self.mge['lum_s'][i]) ** 2)
        weights /= weights.sum(axis=1)[:, np.newaxis]
        
        self.x_kappa = x[weights.argmax(axis=0)]
        self.x_kappa[self.mge['lum_s'].argmin()] = 0.
        self.x_kappa[self.mge['lum_s'].argmax()] *= 10

    @property
    def parameters(self):
        if self._parameters is None:
            self._parameters = super(AnalyticalProfiles, self).parameters
            _ = self._parameters.pop('mlr')
            self._parameters.update({'mlr_0': u.dimensionless_unscaled, 'mlr_t': u.dimensionless_unscaled,
                                     'mlr_inf': u.dimensionless_unscaled, 'r_mlr': u.arcsec,
                                     'kappa_x': u.dimensionless_unscaled, 'kappa_y': u.dimensionless_unscaled,
                                     'r_kappa': u.arcsec})
        return self._parameters

    @property
    def parameter_labels(self):
        labels = {}
        for row in self.initials:
            name = row['name']
            latex_string = row['init'].unit.to_string('latex')
            if name == 'd':
                labels[name] = r'$d/${0}'.format(latex_string)
            elif len(name) > 3 and name[:3] == 'mlr':
                suffix = name[4:]
                if suffix == 'inf':
                    suffix = r'\infty'
                labels[name] = r'$\Upsilon_{{\rm {0}}}/{{\rm M_\odot}}\,{{\rm L_\odot^{{-1}}}}$'.format(suffix)
            elif name == 'barq':
                labels[name] = r'$\bar{q}$'
            elif name == 'kappa_x':
                labels[name] = r'$\kappa_{\rm x}$'
            elif name == 'kappa_y':
                labels[name] = r'$\kappa_{\rm y}$'
            elif name == 'beta':
                labels[name] = r'$\beta$'
            elif name == 'r_kappa':
                labels[name] = r'$r_{{\rm \kappa}}$/{0}'.format(latex_string)
            elif name == 'r_mlr':
                labels[name] = r'$r_{{\rm \Upsilon}}$/{0}'.format(latex_string)
            else:
                labels[row['name']] = r'${0}/${1}'.format(row['name'], latex_string)
        return labels

    def fetch_parameters(self, values, return_rkappa=False):

        parameters = super(AnalyticalProfiles, self).fetch_parameters(values)

        _x = self.x_mlr/parameters.pop('r_mlr')
        parameters['mlr'] = (
            parameters.pop('mlr_0')*(1.-_x) + 2.*parameters.pop('mlr_t')*_x + parameters.pop('mlr_inf')*_x*(_x-1.))/(
                1.+_x**2)

        rkappa = parameters.pop('r_kappa')
        _x = (self.x_kappa / rkappa).si
        kappa_max = np.sqrt(parameters['kappa_x']**2 + parameters['kappa_y']**2)
        parameters['kappa'] = 2.* kappa_max *_x / (1. + _x**2)
        
        if return_rkappa:
            return parameters, rkappa
        
        return parameters

    def lnprior(self, values):
        p = 0
        
        # add additional checks for parameters of radial profiles
        for parameter, value in dict(zip(self.fitted_parameters, values)).items():

            # recover unit
            i = [init['name'] for init in self.initials].index(parameter)
            try:
                v = u.Quantity(value, unit=self.initials[i]['init'].unit)
            except u.core.UnitTypeError:
                v = u.Dex(value, unit=self.initials[i]['init'].unit)

            if parameter in ['mlr_0', 'mlr_t'] and (v.value <= 0.1 or v.value > 100):
                return -np.inf

            elif parameter == 'mlr_inf':     
                my_mean = 3.5
                my_std = 1.0

                if value < 0: 
                    return -np.inf
                else:
                    p0 = stats.norm.logpdf(value, loc=my_mean, scale=my_std)
                    # print('log-prior mlr_inf', p0)
                    p = p0 + p

            elif parameter == 'r_mlr' and not (self.mge['mass_s'].min() < v < self.mge['mass_s'].max()):
                return -np.inf

            elif parameter == 'r_kappa' and not (self.mge['lum_s'].min() < v < self.mge['lum_s'].max()):
                return -np.inf

        pradial = p 
        paxis = super(AnalyticalProfiles, self).lnprior(values=values)
        
        #print('pradial', pradial)
        #print('paxis', paxis)
        
        return pradial + paxis
