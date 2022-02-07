import logging
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from astropy import units as u
from astropy.visualization import quantity_support


logger = logging.getLogger(__name__)


class ProfilePlot(object):

    default_style = {'ls': 'None', 'lw': 1.6, 'c': 'g', 'marker': 'o', 'mew': 1.6, 'ms': 6, 'mec': 'g',
                     'mfc': 'g', 'zorder': 2}

    def __init__(self, figure=None):

        quantity_support()

        if figure is None:
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
