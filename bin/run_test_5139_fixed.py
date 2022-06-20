#! /usr/bin/env python3
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import QTable

from mcmc_dynamics.analysis import ModelFit, ConstantFit
from mcmc_dynamics.utils.plots import ProfilePlot
from mcmc_dynamics.utils.files import DataReader
from mcmc_dynamics.utils import coordinates
from astropy.table import Table
# from mcmc_dynamics.parameter import Parameter


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test MCMC code using mock data created on-the-fly.')
    parser.add_argument('-n', '--nstars', type=int, default=500, help='The number of mock stars.')
    parser.add_argument('-r', '--rmax', type=float, default=5.0, help='Maximum data radius relative to scale radius.')
    parser.add_argument('--vsigma', type=float, default=0.5, help='Ratio between max. rotation and dispersion.')
    parser.add_argument('--errscale', type=float, default=0.1, help='Ratio between avg. uncertainty and dispersion.')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed to initialize random-number generator.')

    args = parser.parse_args()
    print('hello')
    # DATA CREATION
    logging.basicConfig(level=logging.INFO)
    logger.info('Creating input data ...')
    np.random.seed(args.seed)

    v_sys = 235*u.km/u.s
    r_peak = 60.*u.arcsec
    a = 30.*u.arcsec
    theta_0 =  2.094395*u.rad#2.*np.pi*np.random.random()*u.rad
    sigma_max = (5. + 10.*np.random.random())*u.km/u.s
    v_max = args.vsigma*sigma_max

    _data = QTable.read('/Users/ljmu/Desktop/muse/datafiles/ngc5139_combined_velocities_jan22.csv')
    _data.rename_columns(names=('STAR V', 'STAR V err'), new_names=('v', 'verr'))
    idx = (_data['Membership'] > 0.6) & (_data['Mean SNR'] > 8.) & (_data['v'] > 100) & (_data['verr'] < 350)
    _data = _data[idx]
    ra,dec = _data['RA'],_data['Decl']
    center = (201.69184583, -47.47911111) #n08
    # (201.69683333, -47.47956944) #a10
    # (201.69630208, -47.47835389)#n10
    #(201.696718746, -47.479909445555) #kinematic center
    a1 = coordinates.calc_xy_offset(ra, dec, center[0] * u.deg, center[1] * u.deg)
    _data['x'], _data['y'] = a1
    data = DataReader(_data)
    data.compute_polar()
    # data.data['theta'] = data.data['theta'] + 4.712 * u.rad

    # data = DataReader({
    #     'r': r_peak*args.rmax*np.random.uniform(0, 1, size=args.nstars)**0.9,
    #     'theta': u.rad*np.random.uniform(0, 2.*np.pi, size=args.nstars),
    #     'v': np.zeros((args.nstars,), dtype=np.float64)*u.km/u.s,
    #     'verr': np.zeros((args.nstars,), dtype=np.float64)*u.km/u.s})

    x_pa = data.data['r'] * np.sin(data.data['theta'] - theta_0)
    # v_los = v_sys + 2. * (v_max / r_peak) * x_pa / (1. + (data.data['r'] / r_peak) ** 2)
    #
    # sigma_los = sigma_max / (1. + (data.data['r'] / a) ** 2) ** 0.25
    # v_los += sigma_los*np.random.randn(len(data.data['r']))
    #
    # uncertainties = args.errscale*sigma_los*np.random.lognormal(0, 0.5, size=len(data.data['r']))
    #
    # v_los += uncertainties*np.random.randn(len(data.data['r']))
    #
    # data.data['v'] = v_los
    # data.data['verr'] = uncertainties
    logger.info(data.data)

    # FIT IN RADIAL BINS
    logger.info('Analysing kinematics in radial bins ...')
    data.make_radial_bins(nstars=100, dlogr=0.1)

    # prepare container for storing results from analysis in radial bins
    radial_bins = []
    column_names = ('r mean', 'r min', 'r max')

    for i in range(data.data['bin'].max() + 1):
        data_i = data.fetch_radial_bin(i)

        # initialize sampler
        cf = ConstantFit(data_i, parameters=None, background=None)

        # modify function for creating initials for chains
        cf.parameters['sigma_max'].set(min=0, max=100,initials='rng.lognormal(mean={0:.2f}, sigma=0.5, size=n)'.format(np.log(10.)))
        cf.parameters['v_maxx'].set(min=-10, max=10,initials='rng.normal(loc=0, scale=2, size=n)')
        cf.parameters['v_sys'].set(value=230.5,fixed=True)
        # cf.parameters['v_sys'].set(min=0, max=400,initials='rng.normal(loc=232, scale=3, size=n)')
        # cf.parameters['a'].set(initials='rng.normal(loc=15, scale=3, size=n)', min=0, max=300)
        # cf.parameters.add(name='theta_0', value=theta_0, min=0, max=2.*np.pi, fixed=True)
        # cf.parameters.add(name='theta_0', value=theta_0, min=0, max=2. * np.pi,fixed=True)
        # cf.parameters['v_maxy'].set(value=0.0,fixed=True)
        cf.parameters['v_maxy'].set(min=-5, max=5,initials='rng.normal(loc=0, scale=1, size=n)')  # , expr="v_maxx*tan(theta_0)")

        if i == 0:
            cf.parameters.pretty_print()
        sampler = cf(n_walkers=100, n_steps=100, n_threads=1)
        # _ = cf.plot_chain(chain=sampler.chain)
        # _ = cf.create_triangle_plot(chain=sampler.chain, n_burn=50)

        write_table = Table({'chain': sampler.chain, 'lnprob': sampler.lnprobability})
        write_table.write('/Users/ljmu/Desktop/muse/output/mcmc_chains_constant_fixedvsys_n08.fits', format='fits', overwrite=True)
        plt.show()

        results_i = (data_i.data['r'].mean(), data_i.data['r'].min(), data_i.data['r'].max())

        bestfit_values = cf.compute_bestfit_values(chain=sampler.chain, n_burn=50)
        for name, parameter in cf.parameters.items():
            if parameter.fixed:
                continue
            results_i = results_i + (bestfit_values.loc['median'][name],
                                     bestfit_values.loc['uperr'][name],
                                     bestfit_values.loc['loerr'][name])
            if i == 0:
                column_names = column_names + (name + ' median', name + ' high', name + ' low')

        theta_vmax = cf.compute_theta_vmax(chain=sampler.chain, n_burn=50)
        if theta_vmax is not None:
            for name in ['v_max', 'theta_0']:
                results_i = results_i + (theta_vmax.loc['median'][name],
                                         theta_vmax.loc['uperr'][name],
                                         theta_vmax.loc['loerr'][name])
                if i == 0:
                    column_names = column_names + (name + ' median', name + ' high', name + ' low')

        radial_bins.append(results_i)

    radial_profile = QTable(rows=radial_bins, names=column_names)
    print(radial_profile)

    # MODEL FIT
    logger.info('Fitting radial model to data ...')
    mf = ModelFit(data=data, parameters=None)

    # modify initials for some parameters. Radii are initialized using beta-function between min and max radius covered
    # by data
    r_min = data.data['r'].min()
    r_max = data.data['r'].max()
    mf.parameters['sigma_max'].set(min=0, max=100,initials='rng.lognormal(mean={0:.2f}, sigma=0.5, size=n)'.format(np.log(10.)))
    mf.parameters['a'].set(min=0, max=300, initials='300*rng.beta(a=2, b=5, size=n)')
    mf.parameters['v_maxx'].set(min=-10, max=10, initials='rng.normal(loc=0, scale=2, size=n)')
    # mf.parameters.add(name='theta_0', value=theta_0, min=0, max=2.*np.pi, fixed=True)
    # mf.parameters['v_maxy'].set(min=-10, max=10,initials='rng.normal(loc=0, scale=2, size=n)')  # , expr="v_maxx*tan(theta_0)")
    mf.parameters['r_peak'].set(min=0, max=300, initials='300*rng.beta(a=2, b=5, size=n)')
    mf.parameters['v_sys'].set(min=0, max=400,initials='rng.normal(loc=232, scale=3, size=n)')
    mf.parameters['v_maxy'].set(min=-5, max=5, initials='rng.normal(loc=0, scale=1, size=n)')  # , expr="v_maxx*tan(theta_0)")

    # mf.parameters.add(name='theta_0', value=theta_0, min=0, max=2. * np.pi, fixed=True)
    # mf.parameters['v_maxy'].set(value=0.0,fixed=True)
    mf.parameters.pretty_print()

    # run model calculation
    sampler = mf(n_threads=1)
    write_table = data = Table({'chain': sampler.chain, 'lnprob': sampler.lnprobability})
    data.write('/Users/ljmu/Desktop/muse/output/mcmc_chains_model_fixedvsys_n08.fits', format='fits', overwrite=True)
    _ = mf.plot_chain(chain=sampler.chain, lnprob=sampler.lnprobability)
    # _ = mf.create_triangle_plot(chain=sampler.chain, n_burn=100)
    # plt.show()

    radial_model = mf.create_profiles(sampler.chain, n_burn=100)

    # PLOTTING
    logger.info('Plotting the results ...')

    # r_true = np.logspace(-1, 2, 50)*u.arcsec
    # v_rot_true = 2. * (v_max / r_peak) * r_true / (1. + (r_true / r_peak) ** 2)
    # sigma_true = sigma_max / (1. + r_true ** 2 / a ** 2) ** 0.25

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
    #
    # pp.add_rotation_profile(r_true.value, v_rot_true, ls='-', lw=1.5, c='k', marker='None')
    # pp.add_dispersion_profile(r_true.value, sigma_true, ls='-', lw=1.5, c='k', marker='None')
    # pp.add_theta_profile(r_true.value, theta_0*np.ones_like(r_true.value), ls='-', lw=1.5, c='k', marker='None')
    radial_profile.write('/Users/ljmu/Desktop/muse/output/radial_profile_fixedvsys_n08.csv')
    radial_model.write('/Users/ljmu/Desktop/muse/output/radial_model_fixedvsys_n08.csv')
    plt.show()
