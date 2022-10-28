import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from astropy import units as u
from astropy.visualization import quantity_support


logger = logging.getLogger(__name__)
COLORS = pd.DataFrame(np.reshape(['black', '#1b9e77', '#d95f02', '#7570b3'], (1, 4)),
                      columns=['OVERALL', 'P1', 'P2', 'P3'])

MARKERS = pd.DataFrame(np.reshape(['v', 'o', '^', 's'], (1, 4)),
                       columns=['OVERALL', 'P1', 'P2', 'P3'])


class ProfilePlot(object):

    default_style = {'ls': 'None', 'lw': 1.6, 'c': 'g', 'marker': 'o', 'mew': 1.6, 'ms': 6, 'mec': 'g',
                     'mfc': 'g', 'zorder': 2}

    def __init__(self, figure=None, ax_rot=None, ax_disp=None, ax_theta=None):

        quantity_support()

        if ax_rot is not None and ax_disp is not None and ax_theta is not None:
            self.ax_rot = ax_rot
            self.ax_theta = ax_theta
            self.ax_disp = ax_disp

        elif figure is None:
            self.figure = plt.figure(figsize=(168. / 25.4, 110 / 25.4))

            gs = gridspec.GridSpec(2, 2)
            gs.update(hspace=0.04, top=0.98, bottom=0.18)

            self.ax_rot = self.figure.add_subplot(gs[0, 0])
            self.ax_theta = self.figure.add_subplot(gs[:, 1], polar=True)
            self.ax_disp = self.figure.add_subplot(gs[1, 0], sharex=self.ax_rot)

        else:
            assert len(figure.axes) == 3, 'No. of axes in provided figure instance != 3.'
            self.figure = figure

            self.ax_rot = figure.axes[0]
            self.ax_theta = figure.axes[1]
            self.ax_disp = figure.axes[2]

        self.ax_rot.set_xscale('log', base=10)
        self.ax_rot.set_xticks(np.logspace(-1, 2, 4))
        self.ax_rot.xaxis.tick_top()
        self.ax_rot.xaxis.set_ticks_position('both')
        self.ax_rot.set_ylabel(r"$v_\mathrm{rot}\ [\mathrm{km/s}$]", fontsize=16)

        # no labels for radial axis, use same upper limit as for rotation plot
        self.ax_theta.set_yticklabels([])

        # make sure north (=0 degrees) is up, use directions as labels
        self.ax_theta.set_xlabel(r"$\theta_\mathrm{0}$", fontsize=18)
        self.ax_theta.set_theta_zero_location('E')
        labels = [r'${\rm W}$', '', r'${\rm N}$', '', r'${\rm E}$', '', r'${\rm S}$', '']
        self.ax_theta.set_thetagrids(np.arange(0, 360, 45), labels=labels, fontsize=16)

        self.ax_disp.set_xlabel(r"$r/\mathrm{arcsec}$", fontsize=16)
        self.ax_disp.set_ylabel(r"$\sigma_\mathrm{r}\ [\mathrm{km/s}$]", fontsize=16)

    def add_dispersion_profile(self, x, y, xerr=None, yerr=None, fill_between=False, **kwargs):

        x = self._convert_values(x, u.arcsec, name='x')
        y = self._convert_values(y, u.km/u.s, name='y')
        xerr = self._convert_values(xerr, default_unit=u.arcsec, name='xerr')
        yerr = self._convert_values(yerr, default_unit=u.km/u.s, name='yerr')

        for key, value in self.default_style.items():
            if key not in kwargs:
                kwargs[key] = value

        _yerr = yerr if not fill_between else None

        self.ax_disp.errorbar(x, y, xerr=xerr, yerr=_yerr, **kwargs)

        if yerr is not None and fill_between:
            if np.ndim(yerr) == 2:
                ymin = np.asarray(y) - np.asarray(yerr[0])
                ymax = np.asarray(y) + np.asarray(yerr[1])
            else:
                ymin = np.asarray(y) - np.asarray(yerr)
                ymax = np.asarray(y) + np.asarray(yerr)

            c = kwargs.pop('c', self.default_style['c'])
            self.ax_disp.fill_between(x, ymax, ymin, linestyle='None', color=c, alpha=0.4)

    def add_rotation_profile(self, x, y, xerr=None, yerr=None, fill_between=False, **kwargs):

        x = self._convert_values(x, u.arcsec, name='x')
        y = self._convert_values(y, u.km/u.s, name='y')
        xerr = self._convert_values(xerr, default_unit=u.arcsec, name='xerr')
        yerr = self._convert_values(yerr, default_unit=u.km/u.s, name='yerr')

        for key, value in self.default_style.items():
            if key not in kwargs:
                kwargs[key] = value

        _yerr = yerr if not fill_between else None

        self.ax_rot.errorbar(x, y, xerr=xerr, yerr=_yerr, **kwargs)

        if yerr is not None and fill_between:
            if np.ndim(yerr) == 2:
                ymin = np.asarray(y) - np.asarray(yerr[0])
                ymax = np.asarray(y) + np.asarray(yerr[1])
            else:
                ymin = np.asarray(y) - np.asarray(yerr)
                ymax = np.asarray(y) + np.asarray(yerr)

            c = kwargs.pop('c', self.default_style['c'])
            self.ax_rot.fill_between(x, ymax, ymin, linestyle='None', color=c, alpha=0.4)

    def add_theta_profile(self, x, y, yerr=None, **kwargs):

        x = self._convert_values(x, u.arcsec, name='x')
        y = self._convert_values(y, u.rad, name='y')
        yerr = self._convert_values(yerr, u.rad, name='yerr')

        for key, value in self.default_style.items():
            if key not in kwargs:
                kwargs[key] = value

        self.ax_theta.plot(y, np.log10(x), **kwargs)

        # if provided, show errorbars, make sure they follow curvature of plot
        if yerr is not None:
            c = kwargs.pop('c', self.default_style['c'])
            lw = kwargs.pop('lw', self.default_style['lw'])

            for i, (th, _r) in enumerate(zip(y, np.log10(x))):
                n_segments = max(6, int((yerr[1][i] + yerr[0][i]) / 0.1))
                local_theta = np.linspace(-yerr[0][i], yerr[1][i], n_segments) + th
                local_r = np.ones(n_segments) * _r
                self.ax_theta.plot(local_theta, local_r, color=c, marker='', lw=lw)

    def add_scale_radius(self, r, **kwargs):
        r = self._convert_values(r, u.arcsec, name='r_hl')

        ls = kwargs.pop('ls', '--')
        lw = kwargs.pop('lw', 1.6)
        c = kwargs.pop('c', '0.5')

        for ax in [self.ax_rot, self.ax_disp]:
            ax.axvline(x=r, ls=ls, lw=lw, c=c, **kwargs)

    @staticmethod
    def _convert_values(values, default_unit, name='x'):

        if values is None:
            return None
        values = u.Quantity(values)
        if values.unit.is_unity():
            values *= default_unit
            logger.warning('No unit for {0}-coordinates provided. Assuming {1}.'.format(name, values.unit))
        else:
            try:
                values = values.to(default_unit)
            except u.UnitConversionError:
                logger.warning('Cannot convert {0}-coordinates values to {1}.'.format(name, default_unit))
        return values.value


class MultiProfilePlot(object):
    def __init__(self, num_profiles, figure=None, labels=None, title=None,
                 axes_rot=None, axes_disp=None, axes_theta=None):

        self.num_profiles = num_profiles

        self.axes_rot, self.axes_disp, self.axes_theta = [], [], []
        corr = 3 / num_profiles

        if axes_rot is not None and axes_disp is not None and axes_theta is not None:
            self.axes_rot = axes_rot
            self.axes_disp = axes_disp
            self.axes_theta = axes_theta
        else:
            if figure is None:
                figure = plt.figure(figsize=(2*168. / 25.4, 2*num_profiles * 30. / 25.4))

            gs = gridspec.GridSpec(num_profiles, 5)
            gs.update(hspace=0.04, wspace=0.04, top=1-(0.08 * corr), bottom=0.10 * corr, left=0.06, right=0.925)

            for i in range(self.num_profiles):
                figure.add_subplot(gs[i, :2])
                figure.add_subplot(gs[i, 2], polar=True)
                figure.add_subplot(gs[i, 3:], sharex=figure.axes[0])

                self.axes_rot.append(figure.axes[i * 3 + 0])
                self.axes_disp.append(figure.axes[i * 3 + 2])
                self.axes_theta.append(figure.axes[i * 3 + 1])

        self.figure = figure
        if self.num_profiles > 1:
            tick_top = True
        else:
            tick_top = False

        pps = []
        for i in range(self.num_profiles):

            ax_rot = self.axes_rot[i]
            ax_disp = self.axes_disp[i]
            ax_theta = self.axes_theta[i]

            pp = ProfilePlot(figure=None, ax_rot=ax_rot, ax_disp=ax_disp, ax_theta=ax_theta)

            pp.ax_rot.set_xscale('log', base=10)
            pp.ax_rot.set_xticks(np.logspace(-1, 2, 4))
            pp.ax_rot.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True,
                                  labelbottom=False, labeltop=False, labelleft=True, labelright=False,
                                  which='both')
            pp.ax_disp.tick_params(axis='both', direction='in', top=True, bottom=True, left=True, right=True,
                                   labelbottom=False, labeltop=False, labelleft=False, labelright=True,
                                   which='both')
            if i == 0:
                pp.ax_rot.tick_params(labeltop=tick_top)
                pp.ax_disp.tick_params(labeltop=tick_top)
            pp.ax_rot.set_ylabel('')
            pp.ax_disp.set_ylabel('')

            if i == self.num_profiles - 1:
                pp.ax_rot.tick_params(labelbottom=True)
                pp.ax_disp.tick_params(labelbottom=True)
                pp.ax_rot.set_xlabel(r"$r\,[\mathrm{arcsec}]$", fontsize=16)
                pp.ax_disp.set_xlabel(r"$r\,[\mathrm{arcsec}]$", fontsize=16)

            # no labels for radial axis, use same upper limit as for rotation plot
            pp.ax_theta.set_yticklabels([])

            # make sure north (=0 degrees) is up, use directions as labels
            pp.ax_theta.set_theta_zero_location('E')
            theta_labels = [r'${\rm W}$', '', r'${\rm N}$', '', r'${\rm E}$', '', r'${\rm S}$', '']
            pp.ax_theta.set_thetagrids(np.arange(0, 360, 45), labels=theta_labels, fontsize=16)
            pp.ax_theta.tick_params(pad=-20)

            if labels is not None:
                label = labels[i]
                pp.ax_rot.annotate(label, xy=(0.05, 0.9), xycoords='axes fraction',
                                   size=14, ha='left', va='top',
                                   bbox=dict(boxstyle='round', fc='w', alpha=0.5))

                c = COLORS[label].values[0]
                m = MARKERS[label].values[0]
                pp.default_style = {'ls': 'None', 'lw': 1.6, 'c': c, 'marker': m, 'mew': 1.6, 'ms': 6, 'mec': c,
                                    'mfc': c, 'zorder': 2}

            pps.append(pp)
        self.pps = pps

        if axes_rot is None or axes_disp is None or axes_theta is None:
            plt.text(x=0.015, y=0.5 - 0.05 * corr, s=r'$v_\mathrm{rot}\ [\mathrm{km/s}$]', fontsize=16, rotation=90,
                     transform=figure.transFigure)
            plt.text(x=0.96, y=0.5 - 0.05 * corr, s=r'$\sigma_\mathrm{r}\ [\mathrm{km/s}$]', fontsize=16, rotation=90,
                     transform=figure.transFigure)

        if title is not None:
            plt.text(0.45, 1-(0.05 * corr), title, transform=figure.transFigure, fontsize=20)

    def add_rotation_profiles(self, model_profiles=None, binned_profiles=None):
        for i in range(self.num_profiles):
            pp = self.pps[i]

            if binned_profiles is not None:
                binned_profile = binned_profiles[i]

                x = binned_profile["r mean"]
                xerr = np.array([binned_profile["r mean"] - binned_profile["r min"],
                                 binned_profile["r max"] - binned_profile["r mean"]]) * x.unit

                vrot = binned_profile['v_max median']
                vrot_err = np.array([binned_profile['v_max low'], binned_profile['v_max high']]) * vrot.unit
                pp.add_rotation_profile(x, vrot, xerr=xerr, yerr=vrot_err)

            if model_profiles is not None:
                model_profile = model_profiles[i]
                pp.add_rotation_profile(model_profile['r'], model_profile['v_rot'],
                                        yerr=[model_profile['v_rot'] - model_profile['v_rot_lower_1s'],
                                              model_profile['v_rot_upper_1s'] - model_profile['v_rot']],
                                        ls='-', lw=1.6, alpha=0.5, marker='None', fill_between=True)

    def add_theta_profiles(self, model_profiles=None, binned_profiles=None, ana_best_fit_params_list=None):
        for i in range(self.num_profiles):
            pp = self.pps[i]
            if binned_profiles is not None:
                binned_profile = binned_profiles[i]

                x = binned_profile["r mean"]

                theta = binned_profile['theta_0 median']
                theta_err = np.array([binned_profile['theta_0 low'], binned_profile['theta_0 high']]) * theta.unit
                pp.add_theta_profile(x, theta, yerr=theta_err)

            if model_profiles is not None and ana_best_fit_params_list is not None:
                ana_best_fit_params = ana_best_fit_params_list[i]
                model_profile = model_profiles[i]
                theta_0, theta_0_uperr, theta_0_loerr = ana_best_fit_params['theta_0']
                radii = model_profile['r'].to(u.arcsec).value
                theta_0_arr = theta_0.value * np.ones(radii.shape) * theta_0.unit

                pp.ax_theta.plot(theta_0_arr, np.log10(radii), c=pp.default_style['c'])
                theta_range = np.linspace(theta_0 - theta_0_loerr, theta_0 + theta_0_uperr, 100)
                pp.ax_theta.fill_between(theta_range,
                                         min(np.log10(radii)), max(np.log10(radii)),
                                         facecolor=pp.default_style['mfc'], alpha=0.5)

    def add_dispersion_profiles(self, model_profiles=None, binned_profiles=None):
        for i in range(self.num_profiles):
            pp = self.pps[i]
            if binned_profiles is not None:
                binned_profile = binned_profiles[i]

                x = binned_profile["r mean"]
                xerr = np.array([binned_profile["r mean"] - binned_profile["r min"],
                                 binned_profile["r max"] - binned_profile["r mean"]]) * x.unit

                sigma = binned_profile['sigma_max median']
                sigma_err = np.array([binned_profile['sigma_max low'], binned_profile['sigma_max high']]) * sigma.unit
                pp.add_dispersion_profile(x, sigma, xerr=xerr, yerr=sigma_err)

            if model_profiles is not None:
                model_profile = model_profiles[i]
                pp.add_dispersion_profile(model_profile['r'], model_profile['sigma'],
                                          yerr=[model_profile['sigma'] - model_profile['sigma_lower_1s'],
                                                model_profile['sigma_upper_1s'] - model_profile['sigma']],
                                          ls='-', lw=1.6, alpha=0.5, marker='None', fill_between=True)

    def add_scale_radii(self, r):
        for i in range(self.num_profiles):
            pp = self.pps[i]
            pp.add_scale_radius(r)


class MultiProfileChromosomeMapPlot(MultiProfilePlot):
    def __init__(self, num_profiles, labels=None, title=None):
        if num_profiles > 1:
            self.figure = plt.figure(figsize=(2 * 168. / 25.4, 1.5 * num_profiles * 30. / 25.4))
        else:
            self.figure = plt.figure(figsize=(2 * 168. / 25.4, 2.5 * 30. / 25.4))

        self.gs = gridspec.GridSpec(num_profiles, 16)
        corr = 3 / num_profiles
        self.gs.update(hspace=0.04, wspace=0.04, top=1-(0.08 * corr), bottom=0.10 * corr, left=0.06, right=0.95)

        axes_rot, axes_disp, axes_theta = [], [], []
        for i in range(num_profiles):
            self.figure.add_subplot(self.gs[i, 6:10])
            self.figure.add_subplot(self.gs[i, 10:12], polar=True)
            self.figure.add_subplot(self.gs[i, 12:], sharex=self.figure.axes[0])

            axes_rot.append(self.figure.axes[i * 3 + 0])
            axes_disp.append(self.figure.axes[i * 3 + 2])
            axes_theta.append(self.figure.axes[i * 3 + 1])

        MultiProfilePlot.__init__(self, num_profiles, figure=self.figure, labels=labels, title=None,
                                  axes_rot=axes_rot, axes_disp=axes_disp, axes_theta=axes_theta)

        plt.text(x=0.36, y=0.5 - 0.05 * corr, s=r'$v_\mathrm{rot}\ [\mathrm{km/s}$]', fontsize=16, rotation=90,
                 transform=self.figure.transFigure)
        plt.text(x=0.975, y=0.5 - 0.05 * corr, s=r'$\sigma_\mathrm{r}\ [\mathrm{km/s}$]', fontsize=16, rotation=90,
                 transform=self.figure.transFigure)

        if title is not None:
            plt.text(0.615, 1-(0.05 * corr), title, transform=self.figure.transFigure, fontsize=20)

        self.ax_cmap = None

    def add_cmap(self, fl_cmap):
        self.figure.add_subplot(self.gs[:, :5])
        self.ax_cmap = self.figure.axes[-1]
        self.ax_cmap.set_xlabel(r'$\Delta_{\mathrm{F275W}-\mathrm{F814W}}$', fontsize=16)
        self.ax_cmap.set_ylabel(r'${\Delta} \mathrm{C}_{\mathrm{F275W}-2\cdot \mathrm{F336W}+\mathrm{F438W}}}$',
                                fontsize=16)

        for fn in fl_cmap:
            pop_name = fn[fn.rfind('POP'): fn.rfind('_restframe.csv')]

            p_name = pop_name.replace('POP', 'P')
            color = COLORS[p_name].values[0]
            marker = MARKERS[p_name].values[0]

            cm = pd.read_csv(fn)
            self.ax_cmap.scatter(cm['dG'], cm['dC'], c=color, marker=marker,
                                 alpha=0.7, label=f'{p_name} (N={len(cm)})')