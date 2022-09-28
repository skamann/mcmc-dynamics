#! /usr/bin/env python3
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from scipy.stats import truncnorm

from mcmc_dynamics.analysis import ModelFit, ConstantFit
from mcmc_dynamics.utils.plots import ProfilePlot
from mcmc_dynamics.utils.files import DataReader


logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test MCMC code using mock data created on-the-fly.')
    parser.add_argument('-n', '--nstars', type=int, default=500, help='The number of mock stars.')
    parser.add_argument('-r', '--rmax', type=float, default=5.0, help='Maximum data radius relative to scale radius.')
    parser.add_argument('--vsigma', type=float, default=0.5, help='Ratio between max. rotation and dispersion.')
    parser.add_argument('--errscale', type=float, default=0.1, help='Ratio between avg. uncertainty and dispersion.')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed to initialize random-number generator.')

    args = parser.parse_args()

    # DATA CREATION
    logging.basicConfig(level=logging.INFO)
    logger.info('Creating input data ...')
    rng = np.random.default_rng(args.seed)

    # Model parameters
    v_sys = 0*u.km/u.s
    r_peak = 60.*u.arcsec
    a = 30.*u.arcsec
    theta_0 = 2.*np.pi*rng.random()*u.rad
    sigma_max = (5. + 10.*rng.random())*u.km/u.s
    v_max = args.vsigma*sigma_max

    # Create mock sample of RA and Dec coordinates around random centre
    sc = SkyCoord(56.345*u.deg, -26.675*u.deg)

    r_max = (r_peak*args.rmax).value
    tn = truncnorm
    tn.random_state = rng
    separation = tn.rvs(a=0, b=r_max, loc=0, scale=r_max/2., size=args.nstars)*r_peak.unit
    position_angle = rng.uniform(-np.pi, np.pi, size=args.nstars)*u.rad
    coordinates = sc.directional_offset_by(position_angle=position_angle, separation=separation)

    data = DataReader({
        'ra': coordinates.ra,
        'dec': coordinates.dec,
        'v': np.zeros((args.nstars,), dtype=np.float64)*u.km/u.s,
        'verr': np.zeros((args.nstars,), dtype=np.float64)*u.km/u.s})

    # Create mock velocity sample
    x_pa = separation * np.sin(position_angle - theta_0)
    v_los = v_sys + 2. * (v_max / r_peak) * x_pa / (1. + (separation / r_peak) ** 2)

    sigma_los = sigma_max / (1. + (separation / a) ** 2) ** 0.25
    v_los += rng.normal(scale=sigma_los, size=args.nstars)*sigma_los.unit

    uncertainties = args.errscale*sigma_los*rng.lognormal(0, 0.5, size=args.nstars)
    v_los += rng.normal(scale=uncertainties, size=args.nstars)*uncertainties.unit

    data.data['v'] = v_los
    data.data['verr'] = uncertainties
    logger.info(data.data)

    # FIT IN RADIAL BINS
    logger.info('Analysing kinematics in radial bins ...')
    data.make_radial_bins(ra_center=sc.ra, dec_center=sc.dec, nstars=50, dlogr=0.1)

    # prepare container for storing results from analysis in radial bins
    radial_bins = []
    column_names = ('r mean', 'r min', 'r max')

    for i in range(data.data['bin'].max() + 1):
        data_i = data.fetch_radial_bin(i)

        # initialize sampler
        cf = ConstantFit(data_i, parameters=None, background=None)

        # modify function for creating initials for chains
        cf.parameters['sigma_max'].set(initials='rng.lognormal(mean={0:.2f}, sigma=0.5, size=n)'.format(np.log(10.)))
        cf.parameters['v_maxx'].set(initials='rng.normal(loc=0, scale=3, size=n)')
        # cf.parameters.add(name='theta_0', value=theta_0, min=0, max=2.*np.pi, fixed=True)
        cf.parameters['v_maxy'].set(initials='rng.normal(loc=0, scale=3, size=n)')  # , expr="v_maxx*tan(theta_0)")
        cf.parameters['ra_center'].set(value=sc.ra, fixed=True)
        cf.parameters['dec_center'].set(value=sc.dec, fixed=True)

        if i == 0:
            cf.parameters.pretty_print()
        sampler = cf(n_walkers=100, n_steps=100, n_threads=1)
        # _ = cf.plot_chain(chain=sampler.chain)
        # _ = cf.create_triangle_plot(chain=sampler.chain, n_burn=50)
        # plt.show()

        r_in_bin = separation[data.data['bin'] == i]
        results_i = (r_in_bin.mean(), r_in_bin.min(), r_in_bin.max())

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
    r_min = separation.min().to(u.arcsec)
    r_max = separation.max().to(u.arcsec)
    mf.parameters['sigma_max'].set(initials='rng.lognormal(mean={0:.2f}, sigma=0.5, size=n)'.format(np.log(10.)))
    mf.parameters['a'].set(
        min=r_min, max=r_max, initials='{0}*rng.beta(a=2, b=5, size=n) + {1}'.format((r_max-r_min).value, r_min.value))
    mf.parameters['v_maxx'].set(initials='rng.normal(loc=0, scale=3, size=n)')
    # mf.parameters.add(name='theta_0', value=theta_0, min=0, max=2.*np.pi, fixed=True)
    mf.parameters['v_maxy'].set(initials='rng.normal(loc=0, scale=3, size=n)')  # , expr="v_maxx*tan(theta_0)")
    mf.parameters['r_peak'].set(
        min=r_min, max=r_max, initials='{0}*rng.beta(a=2, b=5, size=n) + {1}'.format((r_max-r_min).value, r_min.value))
    mf.parameters['ra_center'].set(value=sc.ra, fixed=True)
    mf.parameters['dec_center'].set(value=sc.dec, fixed=True)
    mf.parameters.pretty_print()

    # run model calculation
    sampler = mf(n_threads=1)

    # _ = mf.plot_chain(chain=sampler.chain, lnprob=sampler.lnprobability)
    # _ = mf.create_triangle_plot(chain=sampler.chain, n_burn=100)
    # plt.show()

    radial_model = mf.create_profiles(sampler.chain, n_burn=100)

    # PLOTTING
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
    pp.add_theta_profile(r_true.value, theta_0*np.ones_like(r_true.value), ls='-', lw=1.5, c='k', marker='None')

    plt.show()
