#! /usr/bin/env python3
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import table, units as u
import tqdm
import json
import time

from mcmc_dynamics.analysis import ModelFit, ConstantFit
from mcmc_dynamics.analysis.runner import Runner
from mcmc_dynamics.analysis.cjam import Axisymmetric, AnalyticalProfiles
from mcmc_dynamics.background import SingleStars
from mcmc_dynamics.utils.plots import ProfilePlot
from mcmc_dynamics.utils.files import DataReader, MgeReader

import logging


def get_mge(filename):
    # read in the tracer density MGE from the example and add appropriate units
    _mge = table.Table.read(filename, format='ascii.ecsv')

    _mge['q'] = 0.9

    mge_lum = MgeReader(_mge, lum=True)

    _mge['i'] = _mge['i'] * u.solMass / u.solLum
    mge_mass = MgeReader(_mge, lum=False)

    return mge_lum, mge_mass


def get_observed_data(filename, v_sys):
    params = pd.read_csv(filename)

    logging.info('Assuming mean velocity of {}'.format(v_sys))

    data = DataReader({
        'x': params['x'].values * u.arcmin,
        'y': params['y'].values * u.arcmin,
        'v': params['STAR V'].values * u.km / u.s - v_sys,
        'verr': params['STAR V err'].values * u.km / u.s,
        'pmember': params['Membership']})

    return params, data


def make_radial_plots(runner, chain, data, background, initials, run_number, n_burn):
    radial_model = runner.create_profiles(chain, n_burn=n_burn, n_threads=12, n_samples=20, filename="radial_profiles_{}".format(run_number))

    radial_profile = table.QTable()
    data.make_radial_bins(nstars=100, dlogr=0.1)
    for column in ['r mean', 'r min', 'r max']:
        radial_profile[column] = table.QTable.Column([], unit=data.data['r'].unit)
    for parameter in initials:
        if not parameter['fixed']:
            for column in ['median', 'high', 'low']:
                radial_profile['{0} {1}'.format(parameter['name'], column)] = table.QTable.Column(
                    [], unit=parameter['init'].unit)

    for i in range(data.data['bin'].max() + 1):
        data_i = data.fetch_radial_bin(i)

        results_i = [data_i.data['r'].mean(), data_i.data['r'].min(), data_i.data['r'].max()]

        cf = ConstantFit(data_i, initials=initials, background=background)
        sampler = cf(n_walkers=64, n_steps=100)
        results = cf.compute_bestfit_values(chain=sampler.chain, n_burn=50)

        k = 0
        for parameter in cf.initials:
            if parameter['fixed']:
                continue
            name = parameter['name']
            results_i.extend([results.loc['median'][name], results.loc['uperr'][name], results.loc['loerr'][name]])
            k += 1

        radial_profile.add_row(results_i)

    radial_profile = table.QTable(radial_profile)
    radial_profile.write("binned_radial_profiles_{}".format(run_number), format='ascii.ecsv', overwrite=True)

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

    # pp.add_rotation_profile(r_true.value, v_rot_true, ls='-', lw=1.5, c='k', marker='None')
    # pp.add_dispersion_profile(r_true.value, sigma_true, ls='-', lw=1.5, c='k', marker='None')

    plt.savefig('dispersion_{}.png'.format(run_number))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chain', help='.pkl file with MCMC chains', type=str)
    parser.add_argument('--config', help='json file with config data', type=str)
    parser.add_argument('--restart', help='set to restart the given chain', action="store_true")
    parser.add_argument('--plot', help='only create diagnostic plots for a given chain', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = json.load(open(args.config))
    run_number = int(time.time())

    pos = None
    if args.chain:
        logging.info('Using stored chain {} but with a new run number: {}'.format(args.chain, run_number))
        pos = Runner.read_final_chain(args.chain)
        chain = Runner.read_chain(args.chain)

    params, data = get_observed_data(config['filename_params'], config['v_sys']*u.km/u.s)
    mge_lum, mge_mass = get_mge(config['filename_mge'])

    # rotation angle determined with get_simple_rotation.py
    # data = data.rotate(config['theta_0'])

    initials = json.load(open(config['filename_initials']))['parameters']
    background_data = table.Table.read(config['filename_background'], format='ascii.commented_header',
                                       guess=False, header_start=96)
    background = SingleStars(v=background_data['Vr']*u.km/u.s - config['v_sys'] * u.km/u.s)

    axisym = AnalyticalProfiles(data, mge_mass=mge_mass, mge_lum=mge_lum,
                                initials=initials, background=background, seed=config['seed'])

    if not args.plot:
        logging.info('Starting to run MCMC chain ...')
        sampler = axisym(n_walkers=config['n_walkers'], n_steps=config['n_steps'], n_out=config['n_out'],
                         n_threads=config['n_threads'], plot=True, prefix=str(run_number), pos=pos)

    current_chain = chain if args.plot else sampler.chain
    
    try:
        old_run_number = args.chain[:args.chain.find("_")]
        logging.info("Old run number: {}".format(old_run_number))
        lnprob_file = "{}_lnprob.pkl".format(old_run_number)
        _lnprob = axisym.read_chain(lnprob_file)
    except FileNotFoundError:
        _lnprob = None
        
    axisym.plot_chain(current_chain, filename='cjam_chains_{}.png'.format(run_number), lnprob=_lnprob)

    try:
        logging.info('Creating corner plot ...')
        axisym.create_triangle_plot(current_chain, n_burn=config['n_burn'], filename='cjam_corner_{}.png'.format(run_number), quantiles=[0.16,0.5, 0.84], show_titles=True)
    except Exception as e:
        logging.warning(e)

    initials = [{'name': 'v_sys', 'init': config['v_sys'] * u.km/u.s, 'fixed': True},
                {'name': 'sigma_max', 'init': config['sigma_max'] * u.km/u.s, 'fixed': False},
                {'name': 'v_max', 'init': config['v_max'] * u.km/u.s, 'fixed': False},
                {'name': 'theta_0', 'init': config['theta_0'] * u.rad, 'fixed': False}]

    logging.info('Creating profile plots ... ')
    make_radial_plots(runner=axisym, chain=current_chain, data=data, background=background, initials=initials, run_number=run_number, n_burn=config['n_burn'])
