#! /usr/bin/env python3
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np

from scipy import stats
from astropy import units as u
from astropy.table import QTable

from mcmc_dynamics.analysis import ModelFit, ConstantFit
from mcmc_dynamics.parameter import Parameter, Parameters
from mcmc_dynamics.utils.plots import ProfilePlot
from mcmc_dynamics.utils.files import DataReader


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test MCMC code using mock data created on-the-fly.')
    parser.add_argument('-n', '--nstars', type=int, default=300, help='The number of mock stars.')
    parser.add_argument('-r', '--rmax', type=float, default=2.0, help='Maximum data radius relative to scale radius.')
    parser.add_argument('--vsigma', type=float, default=0.5, help='Ratio between max. rotation and dispersion.')
    parser.add_argument('--errscale', type=float, default=0.1, help='Ratio between avg. uncertainty and dispersion.')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed to initialize random-number generator.')

    args = parser.parse_args()

    # logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logger.info('Creating input data ...')
    np.random.seed(args.seed)

    v_sys = 0*u.km/u.s
    r_peak = 60.*u.arcsec
    a = 30.*u.arcsec
    theta_0 = 2.*np.pi*np.random.random()*u.rad
    sigma_max = (5. + 10.*np.random.random())*u.km/u.s
    v_max = args.vsigma*sigma_max

    data = DataReader({
        'r': r_peak*args.rmax*np.random.uniform(0, 1, size=args.nstars)**0.9,
        'theta': u.rad*np.random.uniform(0, 2.*np.pi, size=args.nstars),
        'v': np.zeros((args.nstars,), dtype=np.float64)*u.km/u.s,
        'verr': np.zeros((args.nstars,), dtype=np.float64)*u.km/u.s})

    x_pa = data.data['r'] * np.sin(data.data['theta'] - theta_0)
    v_los = v_sys + 2. * (v_max / r_peak) * x_pa / (1. + (x_pa / r_peak) ** 2)

    sigma_los = sigma_max / (1. + (data.data['r'] / a) ** 2) ** 0.25
    v_los += sigma_los*np.random.randn(args.nstars)

    uncertainties = args.errscale*sigma_los*np.random.lognormal(0, 0.5, size=args.nstars)

    v_los += uncertainties*np.random.randn(args.nstars)

    data.data['v'] = v_los
    data.data['verr'] = uncertainties
    logger.info(data.data)

    logger.info('Analysing kinematics in radial bins ...')
    data.make_radial_bins(nstars=50, dlogr=0.1)

    parameters = Parameters(usersyms={'norm': stats.norm})
    parameters.add(name='v_sys', value=v_sys, fixed=True)
    parameters.add(name='sigma_max', value=sigma_max, fixed=False, min=0, initials='norm(loc=5, scale=1).rvs')
    parameters.add(name='v_maxx', value=0.5*v_max, fixed=False,
                   initials='norm(loc={0}, scale=1).rvs'.format(0.5*v_max.value))
    parameters.add(name='v_maxy', value=0.5*v_max, fixed=False,
                   initials='norm(loc={0}, scale=1).rvs'.format(0.5*v_max.value))
    parameters.pretty_print()

    # initials = [{'name': 'v_sys', 'init': v_sys, 'fixed': True},
    #             {'name': 'sigma_max', 'init': sigma_max, 'fixed': False},
    #             {'name': 'v_maxx', 'init': 0.5*v_max, 'fixed': False},
    #             {'name': 'v_maxy', 'init': 0.5*v_max, 'fixed': False}]

    # create table for storing results from analysis in radial bins
    radial_profile = QTable()
    for column in ['r mean', 'r min', 'r max']:
        radial_profile[column] = QTable.Column([], unit=data.data['r'].unit)
    for name, parameter in parameters.items():
        if not parameter.fixed:
            for column in ['median', 'high', 'low']:
                radial_profile['{0} {1}'.format(name, column)] = QTable.Column([], unit=parameter.unit)
    for parameter_name, parameter_unit in {'v_max': u.km/u.s, 'theta_0': u.rad}.items():
        for column in ['median', 'high', 'low']:
            radial_profile['{0} {1}'.format(parameter_name, column)] = QTable.Column([], unit=parameter_unit)

    for i in range(data.data['bin'].max() + 1):
        data_i = data.fetch_radial_bin(i)

        results_i = [data_i.data['r'].mean(), data_i.data['r'].min(), data_i.data['r'].max()]

        cf = ConstantFit(data_i, parameters=parameters, background=None)
        sampler = cf(n_walkers=100, n_steps=100, n_threads=1)
        # cf.plot_chain(chain=sampler.chain)
        # cf.create_triangle_plot(chain=sampler.chain, n_burn=50)
        # plt.show()
        results = cf.compute_bestfit_values(chain=sampler.chain, n_burn=50)

        k = 0
        for name, parameter in cf.parameters.items():
            if parameter.fixed:
                continue
            results_i.extend([results.loc['median'][name], results.loc['uperr'][name], results.loc['loerr'][name]])
            k += 1

        theta_vmax = cf.compute_theta_vmax(chain=sampler.chain, n_burn=50)
        if theta_vmax is not None:
            for name in ['v_max', 'theta_0']:
                results_i.extend(
                    [theta_vmax.loc['median'][name], theta_vmax.loc['uperr'][name], theta_vmax.loc['loerr'][name]])

        radial_profile.add_row(results_i)

    radial_profile = QTable(radial_profile)

    logger.info('Fitting radial model to data ...')

    parameters = Parameters(usersyms={'norm': stats.norm, 'lognorm': stats.lognorm})
    parameters.add(name='v_sys', value=v_sys, min=v_sys-10.*u.km/u.s, max=v_sys+10.*u.km/u.s, fixed=False)
    parameters.add(name='sigma_max', value=sigma_max, fixed=False, min=0, initials='norm(loc=5, scale=1).rvs')
    parameters.add(name='a', value=a, fixed=False, min=0, initials='lognorm(s=1, loc={0}).rvs'.format(a.value))
    parameters.add(name='v_maxx', value=0.5 * v_max, fixed=False,
                   initials='norm(loc={0}, scale=1).rvs'.format(0.5 * v_max.value))
    parameters.add(name='v_maxy', value=0.5 * v_max, fixed=False,
                   initials='norm(loc={0}, scale=1).rvs'.format(0.5 * v_max.value))
    parameters.add(name='r_peak', value=r_peak, fixed=False, min=0, max=10.*u.arcmin)
    parameters.pretty_print()
    # initials = [{'name': 'v_sys', 'init': v_sys, 'fixed': False},
    #             {'name': 'sigma_max', 'init': sigma_max, 'fixed': False},
    #             {'name': 'a', 'init': a, 'fixed': False},
    #             {'name': 'v_maxx', 'init': 0.5*v_max, 'fixed': False},
    #             {'name': 'v_maxy', 'init': 0.5*v_max, 'fixed': False},
    #             {'name': 'r_peak', 'init': r_peak, 'fixed': False}]

    mf = ModelFit(data=data, parameters=parameters)
    sampler = mf(n_threads=1)

    _ = mf.plot_chain(chain=sampler.chain, lnprob=sampler.lnprobability)

    radial_model = mf.create_profiles(sampler.chain, n_burn=50)

    logger.info('Plotting the results ...')

    r_true = np.logspace(-1, 2, 50)*u.arcsec
    v_rot_true = 2. * (v_max / r_peak) * r_true / (1. + (r_true / r_peak) ** 2)
    sigma_true = sigma_max / (1. + r_true ** 2 / a ** 2) ** 0.25

    pp = ProfilePlot()

    x = radial_profile["r mean"]
    xerr = np.array([radial_profile["r mean"] - radial_profile["r min"],
                     radial_profile["r max"] - radial_profile["r mean"]]) * x.unit

    vrot = radial_profile['v_max median']
    vrot_err = np.array([radial_profile['v_max low'], radial_profile['v_max high']]) * vrot.unit
    pp.add_rotation_profile(x, vrot, xerr=xerr, yerr=vrot_err)
    pp.ax_rot.axhline(y=0.0, lw=1.5, c='0.5')

    pp.add_rotation_profile(radial_model['r'], radial_model['v_rot'],
                            yerr=[radial_model['v_rot'] - radial_model['v_rot_lower_1s'],
                                  radial_model['v_rot_upper_1s'] - radial_model['v_rot']],
                            ls='-', lw=1.6, c='g', alpha=0.5, marker='None', fill_between=True)

    theta = radial_profile['theta_0 median']
    theta_err = np.array([radial_profile['theta_0 low'], radial_profile['theta_0 high']]) * theta.unit
    pp.add_theta_profile(x, theta, yerr=theta_err)

    sigma = radial_profile['sigma_max median']
    sigma_err = np.array([radial_profile['sigma_max low'], radial_profile['sigma_max high']]) * sigma.unit
    pp.add_dispersion_profile(x, sigma, xerr=xerr, yerr=sigma_err)

    pp.add_dispersion_profile(radial_model['r'], radial_model['sigma'],
                              yerr=[radial_model['sigma'] - radial_model['sigma_lower_1s'],
                                    radial_model['sigma_upper_1s'] - radial_model['sigma']],
                              ls='-', lw=1.6, c='g', alpha=0.5, marker='None', fill_between=True)

    pp.add_rotation_profile(r_true.value, v_rot_true, ls='-', lw=1.5, c='k', marker='None')
    pp.add_dispersion_profile(r_true.value, sigma_true, ls='-', lw=1.5, c='k', marker='None')

    plt.show()
