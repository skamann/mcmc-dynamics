#! /usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import units as u
from mcmc_dynamics.analysis import model, constant
from mcmc_dynamics.utils.plots import profile_plot
from mcmc_dynamics.utils import radial_profile
from mcmc_dynamics.utils.files import data_reader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Test MCMC code using mock data created on-the-fly.')
    parser.add_argument('-n', '--nstars', type=int, default=300, help='The number of mock stars.')
    parser.add_argument('-r', '--rmax', type=float, default=1.0, help='Maximum data radius relative to scale radius.')
    parser.add_argument('--vsigma', type=float, default=0.5, help='Ratio between max. rotation and dispersion.')
    parser.add_argument('--errscale', type=float, default=0.1, help='Ratio between avg. uncertainty and dispersion.')
    parser.add_argument('-s', '--seed', type=int, default=None, help='Seed to initialize random-number generator.')

    args = parser.parse_args()

    np.random.seed(args.seed)

    v_sys = 0*u.km/u.s
    r_peak = 60.*u.arcsec
    a = 30.*u.arcsec
    theta_0 = 2.*np.pi*np.random.random()*u.rad
    sigma_max = (5. + 10.*np.random.random())*u.km/u.s
    v_max = args.vsigma*sigma_max

    data = data_reader.DataReader({
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
    print(data.data)

    rp = radial_profile.RadialProfile(data.data['r'])
    bin_number = rp(nstars=50, dlogr=0.1)

    radial_profile = []

    for i in range(bin_number.max() + 1):

        in_bin = (bin_number == i)

        results_i = {'logr mean': np.log10(data.data['r'][in_bin].mean()),
                     'logr min': np.log10(data.data['r'][in_bin].min()),
                     'logr max': np.log10(data.data['r'][in_bin].max())}

        data_i = data_reader.DataReader(data.data[in_bin])

        cf = constant.ConstantFit(data_i, fixed={'v_sys': v_sys.value})
        sampler = cf()
        results = cf.compute_bestfit_values(sampler=sampler, n_burn=100)

        for k, parameter in enumerate(cf.fitted_parameters):
            median, high, low = results[k]
            results_i['{0} low'.format(parameter)] = low
            results_i['{0} median'.format(parameter)] = median
            results_i['{0} high'.format(parameter)] = high

        radial_profile.append(results_i)

    radial_profile = pd.DataFrame(radial_profile)

    mf = model.ModelFit(data, fixed={'v_sys': v_sys.value})
    sampler = mf()
    # mf.create_triangle_plot(sampler, n_burn=100, true_values=[v_sys, v_max, r_peak, theta_0, sigma_max, a])
    model = mf.create_profiles(sampler=sampler, n_burn=100)

    r_true = np.logspace(-1, 2, 50)*u.arcsec
    v_rot_true = 2. * (v_max / r_peak) * r_true / (1. + (r_true / r_peak) ** 2)
    sigma_true = sigma_max / (1. + r_true ** 2 / a ** 2) ** 0.25

    pp = profile_plot.ProfilePlot()

    x = radial_profile["logr mean"]
    xerr = [radial_profile["logr mean"] - radial_profile["logr min"],
            radial_profile["logr max"] - radial_profile["logr mean"]]

    vrot = radial_profile['v_max median']
    vrot_err = [radial_profile['v_max low'], radial_profile['v_max high']]
    pp.add_rotation_profile(x, vrot, xerr=xerr, yerr=vrot_err, c='r', mec='r', mfc='r', marker='d')
    pp.add_rotation_profile(np.log10(r_true.value), v_rot_true, ls='-', lw=1.5, c='k', marker='None')

    pp.add_rotation_profile(np.log10(model['r']), model['v_rot'],
                            yerr=[model['v_rot'] - model['v_rot_lower_1s'],
                                  model['v_rot_upper_1s'] - model['v_rot']],
                            ls='-', lw=1.6, c='r', alpha=0.5, marker='None', fill_between=True)

    theta = radial_profile['theta_0 median']
    theta_err = [radial_profile['theta_0 low'], radial_profile['theta_0 high']]
    pp.add_theta_profile(x, theta, yerr=theta_err, marker='d', c='r', mfc='r', mec='r')

    sigma = radial_profile['sigma_max median']
    sigma_err = [radial_profile['sigma_max low'], radial_profile['sigma_max high']]
    pp.add_dispersion_profile(x, sigma, xerr=xerr, yerr=sigma_err, c='r', mec='r', mfc='r', marker='d')
    pp.add_dispersion_profile(np.log10(r_true.value), sigma_true, ls='-', lw=1.5, c='k', marker='None')

    pp.add_dispersion_profile(np.log10(model['r']), model['sigma'],
                              yerr=[model['sigma'] - model['sigma_lower_1s'],
                              model['sigma_upper_1s'] - model['sigma']],
                              ls='-', lw=1.6, c='r', alpha=0.5, marker='None', fill_between=True)

    plt.show()
