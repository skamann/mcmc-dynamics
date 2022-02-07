#! /usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.patches import Ellipse
from astropy import units as u
from astropy import visualization
from astropy.table import Table
from mcmc_dynamics.utils.coordinates import calc_xy_offset
from mcmc_dynamics.utils.morphology import elongation


parser = argparse.ArgumentParser(description="Determine eccentricity and ellipticity of a cluster from its photometry.")
parser.add_argument('photometry', type=str, help="The file containing the photometry to be processed in csv-format.")
parser.add_argument("center", type=u.Quantity, nargs=2, help="Central RA and Dec coordinates used in calculation.")
parser.add_argument("-m", "--magcut", type=float,
                    help="The limiting magnitude level in the chosen passband (see below).")
parser.add_argument('-r', '--radii', type=u.Quantity, nargs='+', help='Inner and outer radii of used annuli.')
parser.add_argument("--ra", type=str, default="RA", help="Column containing right ascension coordinates of sources.")
parser.add_argument("--dec", type=str, default="Decl", help="Column containing declination coordinates of sources.")
parser.add_argument("-f", "--passband", default="F606W",
                    help="The name(s) of the passband(s) used to select the stars.")
parser.add_argument('-o', '--outfilename', help='Filename for storing results.')
parser.add_argument('-p', '--plot', action='store_true', default=False,
                    help='Set this flag to show a plot of the results.')

args = parser.parse_args()

photometry = Table.read(args.photometry, format='ascii.csv')

required_columns = [args.ra, args.dec]
if args.magcut is not None:
    required_columns.append(args.passband)

for column in required_columns:
    if column not in photometry.columns:
        raise IOError('Missing column "{}" in file {}.'.format(column, args.photometry))

dx, dy = calc_xy_offset(photometry[args.ra], photometry[args.dec], ra_center=args.center[0], dec_center=args.center[1])

if args.magcut is not None:
    slc = photometry[args.passband] < args.magcut
else:
    slc = np.ones(len(photometry), dtype=bool)

results = elongation.get_eccentricity_and_pa(dx[slc], dy[slc], bootstrap=True, radii=args.radii)

if args.outfilename is not None:
    results.write(args.outfilename)

if args.plot:
    with visualization.quantity_support():
        fig = plt.figure(figsize=(16, 8))
        gs = gridspec.GridSpec(2, 2)

        ax_data = fig.add_subplot(gs[:, 0], aspect='equal')
        ax_data.scatter(dx[~slc], dy[~slc], marker='o', s=5, alpha=0.5, c='0.5')
        ax_data.scatter(dx[slc], dy[slc], marker='o', s=10, alpha=0.5, c='C0')
        ax_data.plot(0, 0, ls='None', marker='x', mew=2.5, ms=15, c='C3')

        for row in results:
            ell = Ellipse(xy=(0, 0),
                          width=2*row['r_max']*np.sqrt(1.-row['e']**2),
                          height=2*row['r_max'],
                          angle=row['theta'].to('deg').value,
                          linewidth=1.5, edgecolor='C3', facecolor='None')
            ax_data.add_patch(ell)

        ax_theta = fig.add_subplot(gs[0, 1])
        ax_theta.errorbar(results['r_mean'],
                          results['theta'].to(u.deg),
                          xerr=[results['r_mean'] - results['r_min'], results['r_max'] - results['r_mean']],
                          yerr=results['theta_err'].to(u.deg),
                          ls='None', lw=1.5, c='C0', marker='D', mew=1.5, mec='C0', mfc='C0', capsize=3)
        ax_theta.set_ylabel(r'$\Theta_{\rm a}\,[{\rm deg}]$', fontsize=16)

        ax_e = fig.add_subplot(gs[1, 1], sharex=ax_theta)
        ax_e.errorbar(results['r_mean'], results['e'],
                      xerr=[results['r_mean'] - results['r_min'], results['r_max'] - results['r_mean']],
                      yerr=results['e_err'],
                      ls='None', lw=1.5, c='C0', marker='D', mew=1.5, mec='C0', mfc='C0', capsize=3)
        ax_e.set_ylabel(r'$e$', fontsize=16)
        # ax_e.set_xlabel(r'$r\,[{\rm arcmin}]$', fontsize=16)

        fig.tight_layout()
        plt.show()
