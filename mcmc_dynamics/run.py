#! /usr/bin/env python3
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy import table, units as u
from astropy.table import QTable
import tqdm
import json
import time
import sys

import pickle

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
    #params = params[params['Membership'] > 0.7]

    logging.info('Assuming mean velocity of {}'.format(v_sys))

    data_dict = {
        'x': params['x'].values * u.arcmin,
        'y': params['y'].values * u.arcmin,
        'v': params['STAR V'].values * u.km / u.s - v_sys,
        'verr': params['STAR V err'].values * u.km / u.s,
        'pmember': params['Membership']}
        
    data = DataReader(data_dict)

    return params, data

def plot_radial_profiles(radial_model, radial_profile, run_number=None, filename=None):
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

    if filename is not None:
        _filename = filename
    else:
        _filename = 'dispersion_{}.png'.format(run_number)
    plt.savefig(_filename)
    

def generate_radial_data(data, background, initials, run_number, deltas_x=(0,), deltas_y=(0,)): 
    members = data
    members.make_radial_bins(nstars=500, dlogr=0.2)

    # prepare output table
    radial_profile = QTable()
    for column in ['r mean', 'r min', 'r max']:
        radial_profile[column] = QTable.Column([], unit=members.data['r'].unit)
    for parameter in initials:
        if not parameter['fixed']:
            for column in ['median', 'high', 'low']:
                radial_profile['{0} {1}'.format(parameter['name'], column)] = QTable.Column(
                    [], unit=parameter['init'].unit)
                
    if 'v_maxx median' in radial_profile.columns and 'v_maxy median' in radial_profile.columns:
        for column in ['median', 'high', 'low']:
            radial_profile['{0} {1}'.format('v_max', column)] = QTable.Column([], unit=u.km/u.s)
        for column in ['median', 'high', 'low']:
            radial_profile['{0} {1}'.format('theta_0', column)] = QTable.Column([], unit=u.rad)
    #radial_profile['delta_x'] = QTable.Column([], unit=u.arcsec)
    #radial_profile['delta_y'] = QTable.Column([], unit=u.arcsec)
    
    table = []
    samples = []
 
    for offi, (delta_x, delta_y) in enumerate(zip(deltas_x, deltas_y)):
        logging.info("#################---------------------------------------------------######################")
        logging.info("Using offsets {} of {}. delta_x: {:.3f}, delta_y: {:.3f}".format(offi+1, len(deltas_x), delta_x, delta_y))
        logging.info("#################---------------------------------------------------######################")
        
        members.apply_offset(delta_x, delta_y)    
        members.make_radial_bins(nstars=100, dlogr=0.2, force=True)
    
        for i in range(members.data['bin'].max() + 1):
            
            data_i = members.fetch_radial_bin(i)
            #data_i.data.write("binned_{}_{}.csv".format(run_number, i), format='ascii.ecsv', overwrite=True)
            
            results_i = [data_i.data['r'].mean(), data_i.data['r'].min(), data_i.data['r'].max()]
            
            cf = ConstantFit(data_i, initials=initials, background=background)
            sampler = cf(n_walkers=16, n_steps=300)

            results = cf.compute_bestfit_values(chain=sampler.chain, n_burn=100)
            theta_vmax, vmax, theta, sigmas = cf.compute_theta_vmax(chain=sampler.chain, n_burn=100,
                                                            return_samples=True)
            
            samples_df = pd.DataFrame({'theta': theta, 'vmax': vmax, 'sigma': sigmas})            
            samples_df['delta_x'] = delta_x
            samples_df['delta_y'] = delta_y
            samples_df['offsetid'] = offi
            samples_df['binid'] = i
            samples_df['r mean'] = data_i.data['r'].mean()
            
            samples.append(samples_df)
            
            row = {'delta_x': delta_x,
                   'delta_y': delta_y,
                   'offsetid': offi,
                   'binid': i,
                   'r mean': data_i.data['r'].mean(),
                   'r min': data_i.data['r'].min(),
                   'r max': data_i.data['r'].max(),
                   'sigma_max median': results.loc['median']['sigma_max'],
                   'sigma_max high':   results.loc['uperr']['sigma_max'],
                   'sigma_max low':    results.loc['loerr']['sigma_max'], 
                   'v_maxx median':    results.loc['median']['v_maxx'],  
                   'v_maxx high':      results.loc['uperr']['v_maxx'], 
                   'v_maxx low':       results.loc['loerr']['v_maxx'],
                   'v_maxy median':    results.loc['median']['v_maxy'],
                   'v_maxy high':      results.loc['uperr']['v_maxy'],
                   'v_maxy low':       results.loc['loerr']['v_maxy'],
                   'v_max median':     theta_vmax.loc['median']['v_max'],
                   'v_max high':       theta_vmax.loc['uperr']['v_max'],
                   'v_max low':        theta_vmax.loc['loerr']['v_max'],
                   'theta_0 median':   theta_vmax.loc['median']['theta_0'],
                   'theta_0 high':     theta_vmax.loc['uperr']['theta_0'],
                   'theta_0 low':      theta_vmax.loc['loerr']['theta_0'],
                   }
            rowvalues = {}
            for k,v in row.items():
                try:
                    rowvalues[k] = v.value
                except AttributeError:
                    rowvalues[k] = v
                
            table.append(rowvalues)
                
            k = 0
            for parameter in cf.initials:
                if parameter['fixed']:
                    continue
                name = parameter['name']
                results_i.extend([results.loc['median'][name], results.loc['uperr'][name], results.loc['loerr'][name]])
                k += 1

            if theta_vmax is not None:
                for name in ['v_max', 'theta_0']:
                    results_i.extend(
                        [theta_vmax.loc['median'][name], theta_vmax.loc['uperr'][name], theta_vmax.loc['loerr'][name]])#, delta_x, delta_y])
                    
            #print(results_i)
            radial_profile.add_row(results_i)
        
        members.apply_offset(-delta_x, -delta_y)    
        
    radial_profile.write('binned_profile_{}.csv'.format(run_number), format='ascii.ecsv', overwrite=True)
    
    table = pd.DataFrame(table)
    table.to_csv('binned_profile_{}_pd.csv'.format(run_number))
    
    samples = pd.concat(samples, ignore_index=True)
    samples.to_csv('binned_profile_{}_allsamples.csv'.format(run_number))

    return radial_profile


def make_radial_plots(runner, chain, data, background, initials, run_number, n_burn, radial_model=None, radial_profile=None):
    if radial_model is None:
        radial_model = runner.create_profiles(chain, n_burn=n_burn, n_threads=12, n_samples=100, filename="radial_profiles_{}.csv".format(run_number))
    if radial_profile is None:
        radial_profile = generate_radial_data(chain, data, background, initials, run_number, n_burn)
    
    plot_radial_profiles(radial_model=radial_model, radial_profile=radial_profile, run_number=run_number)
    
def make_mlr_plot(runner, chain, n_burn, n_samples=128):
    axisym = runner
    
    # get random set of M/L values from chain
    mlr = [p['mlr'] for p in axisym.sample_chain(chain=chain, n_burn=n_burn, n_samples=n_samples)]

    arcsec2pc = 1/3600/360*2*np.pi * 10000 * u.parsec/u.arcsec
    sigma = (arcsec2pc*axisym.mge_mass.data['s']).to(u.pc)
    intensity = axisym.mge_mass.data['i']
    get_mass = lambda mlr: np.sum(mlr*2.*np.pi*sigma**2*intensity)
    get_meanmlr = lambda mlr: np.sum(mlr*sigma**2*intensity)/np.sum(sigma**2*intensity)
    
    masses = [get_mass(mlr_i).value for mlr_i in mlr]
    means = [get_meanmlr(mlr_i).value for mlr_i in mlr]

    lolim, median, uplim = np.percentile(means, [16, 50, 84])
    print(lolim, median, uplim)
    print('M/L: {0} + {1} - {2} M_sun/L_sun'.format(median, uplim - median, median - lolim))

    lolim, median, uplim = np.percentile(masses, [16, 50, 84])
    print(lolim, median, uplim)
    print('Cluster mass: {0} + {1} - {2} M_sun'.format(median, uplim - median, median - lolim))

    r_mge = np.logspace(-1, 2, 200)*u.arcsec

    mlr_profiles = [axisym.calculate_mlr_profile(mlr_i, radii=r_mge)[1] for mlr_i in mlr]
    lolim, median, uplim = np.percentile(mlr_profiles, [16, 50, 84], axis=0)

    plt.style.use('sciencepaper')
    fig, ax = plt.subplots(1,1, figsize=(5, 2.5))

    color_label = "#555555"
    # core and half-light radius
    ax.axvline(x=(0.15*u.arcmin).to(u.arcsec).value, ls='-', lw=1, c=color_label)
    ax.axvline(x=(0.61*u.arcmin).to(u.arcsec).value, ls='-', lw=1, c=color_label)
    
    ax.text((0.15*u.arcmin).to(u.arcsec).value-0.1, 5-0.25, 'core radius', rotation=90,  horizontalalignment='right', fontdict={'color': color_label})
    ax.text((0.61*u.arcmin).to(u.arcsec).value-1, 5-0.25, 'half-light radius', rotation=90,  horizontalalignment='right', fontdict={'color': color_label})

    lw = 1
    for p in mlr_profiles:
        ax.plot(r_mge, p, ls='-', lw=lw, c='#AAAAAA', alpha=0.2)
    ax.plot(r_mge, median, ls='-', lw=lw, c='#424874', alpha=1)
    ax.plot(r_mge, lolim, ls='-', lw=lw, c='#424874', alpha=1)    
    ax.plot(r_mge, uplim, ls='-', lw=lw, c='#424874', alpha=1)    
    #ax.fill_between(r_mge, lolim, uplim, linewidth=0, facecolor='#', alpha=0.2)

    ax.set_xscale('log', basex=10)

    ax.set_xlabel(r'$r$ [arcsec]')
    ax.set_ylabel(r'$\Upsilon\, [{\rm M_\odot}\,{\rm L_\odot}^{-1}]$')

    fig.tight_layout()
    fig.savefig('ngc6093_mlr_profile.pdf', bbox_inches='tight')
    
def plot_kappas(runner, chain):
    bins = 20 #np.arange(-1, 1, 0.05)
    
    kappas = []
    rkappas = []
    for walker in chain:
        p, rk = runner.fetch_parameters(walker[-1], return_rkappa=True)
        k = p["kappa"]

        rkappas.append(rk.value)
        kappas.append(k)
        
    kappas = np.vstack(kappas)
    rkappas = np.asarray(rkappas)
    
    fig, subs = plt.subplots(nrows=len(k), ncols=1, figsize=(6,12), sharex=True)
    figc, subc = plt.subplots(1,1, figsize=(6,6))
    
    print("comp.\tmin\tmax\tmedian")
    row = "{}\t{:.2f}\t{:.2f}\t{:.2f}"
    for i, k in enumerate(kappas.T.value):
        print(row.format(i, min(k), max(k), np.median(k)))
        _, bins, _ = subs[i].hist(k, alpha=0.5, label="component " + str(i), bins=bins)
        
        subc.scatter(rkappas, k, label="component " + str(i))

    subc.legend()
    for sub in subs:
        sub.legend(loc="upper left")
    fig.savefig("kappas.pdf")
    figc.savefig("corr_kappa.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--chain', help='.pkl file with MCMC chains', type=str)
    parser.add_argument('--config', help='json file with config data', type=str)
    parser.add_argument('--restart', help='set to restart the given chain', action="store_true")
    parser.add_argument('--plot', help='only create diagnostic plots for a given chain', action='store_true')
    parser.add_argument('--name', help='str to use instead of run_number', type=str)
    parser.add_argument('--modelfile', type=str)
    parser.add_argument('--datafile', type=str)
    parser.add_argument('--lnprob_file', type=str)
    parser.add_argument('--allcentres', action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    config = json.load(open(args.config))
    run_number = int(time.time())
    if args.name:
        run_number = args.name

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
    background = SingleStars(v=background_data['Vr']*u.km/u.s - config['v_sys']*u.km/u.s)

    axisym = AnalyticalProfiles(data, mge_mass=mge_mass, mge_lum=mge_lum,
                                initials=initials, background=background, seed=config['seed'])

    if not args.plot:
        logging.info('Starting to run MCMC chain ...')
        sampler = axisym(n_walkers=config['n_walkers'], n_steps=config['n_steps'], n_out=config['n_out'],
                         n_threads=config['n_threads'], plot=True, prefix=str(run_number), pos=pos)

    current_chain = chain if args.plot else sampler.chain
    
    if args.chain:
        try:
            old_run_number = args.chain[:args.chain.find("_")]
            logging.info("Old run number: {}".format(old_run_number))
            lnprob_file = args.lnprob_file
            _lnprob = axisym.read_chain(lnprob_file)
        except FileNotFoundError:
            logging.warn('No file with lnprobs found', lnprob_file)
            _lnprob = None

    _lnprob = _lnprob if args.chain else sampler.lnprobability 

    axisym.plot_chain(current_chain, filename='cjam_chains_{}.png'.format(run_number), lnprob=_lnprob)
    #plot_kappas(axisym, current_chain)
    #logging.info("Plotted kappas.")

    try:
        logging.info('Creating corner plot ...')
        axisym.create_triangle_plot(current_chain, n_burn=config['n_burn'], filename='cjam_corner_{}.png'.format(run_number), quantiles=[0.16,0.5, 0.84], show_titles=True)
    except Exception as e:
        logging.warning(e)

    logging.info('Creating profile plots ... ')
    #make_radial_plots(runner=axisym, chain=current_chain, data=data, background=background, initials=initials, run_number=run_number, n_burn=config['n_burn'])
    
    make_mlr_plot(axisym, current_chain, config['n_burn'])
    logging.info("Plotted M/L profile.")
    
    initials = [{'name': 'v_sys', 'init': 0 * u.km/u.s, 'fixed': True},
                {'name': 'sigma_max', 'init': config['sigma_max'] * u.km/u.s, 'fixed': False},
                {'name': 'v_maxx', 'init': config['v_maxx'] * u.km/u.s, 'fixed': False},
                {'name': 'v_maxy', 'init': config['v_maxy'] * u.km/u.s, 'fixed': False}]

    
    if args.datafile is not None:
        radial_profile = table.QTable.read(args.datafile, format='ascii.ecsv')
    else:
        logging.info("Generating binned data ...")
        parameters = axisym.sample_chain(current_chain, n_burn=config['n_burn'], n_samples=100)
        delta_x = (np.median([p["delta_x"].value for p in parameters]) * parameters[0]["delta_x"].unit, )
        delta_y = (np.median([p["delta_y"].value for p in parameters]) * parameters[0]["delta_y"].unit, )
        
        if args.allcentres:   
            delta_x = [p["delta_x"] for p in parameters]
            delta_y = [p["delta_y"] for p in parameters]
        else:
            logging.info("Using only median centre offset.")        
            logging.info("Accounting for shift in centre: deltax = {:.2f}, delta_y = {:.2f}".format(delta_x, delta_y))

        radial_profile = generate_radial_data(data, background, initials, run_number, deltas_x=delta_x, deltas_y=delta_y)
    
    if args.modelfile is not None:
        logging.info("Reading model file {}".format(args.modelfile))
        radial_model = table.QTable.read(args.modelfile, format='ascii.ecsv')
    else:
        radial_model = axisym.create_profiles(current_chain, n_burn=config['n_burn'], n_threads=50, 
                                              n_samples=100, filename="radial_profiles_{}.csv".format(run_number))       

        
    logging.info("Plotting profiles ...")
    plot_radial_profiles(radial_model=radial_model, radial_profile=radial_profile, run_number=run_number)
