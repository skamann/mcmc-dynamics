# Quickstart

The purpose of MCMC-Dynamics is to fit dynamical models to  observed stellar kinematics data
using maximum-likelihood methods, in particular Markov-Chain Monte Carlo (MCMC).

## Preparing the data

The observed stellar kinematics should be prepared as [astropy QTable](https://docs.astropy.org/en/stable/table/) objects, containing one column per observed quantity
and one row per star. In most cases, the WCS coordinates of the stars (`ra`, `dec`), their
measured velocities (`v`), and the uncertainties of the latter (`verr`) should be provided.
The round brackets contain the column names expected by the code.

MCMC-Dynamics includes the [DataReader](mcmc_dynamics.utils.files.data_reader.DataReader)
class to facilitate the input data processing. The following loads the  MUSE radial velocity data
of M80 published by
[Goettgens et al. (2021)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.507.4788G/abstract) and prepares them for an
analysis with MCMC-Dynamics.

```python
from astropy.table import QTable
data = QTable.read('https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VCNHOR/U1IQHD',
                   format='ascii.tab')
print(data.columns)
# <TableColumns names=('x','y','STAR V','STAR V err','Membership','RA','Decl')>

# rename columns to match expectations of MCMC-Dynamics
data.rename_columns(['RA', 'Decl', 'STAR V', 'STAR V err'], ['ra', 'dec', 'v', 'verr'])

from mcmc_dynamics.utils.files import DataReader
input_data = DataReader(data=data)

# bin data in radial bins around centre, using a minimum of 100 stars and a minimum log-radial range
# of log(d/arcsec) > 0.3
from astropy import units as u
input_data.make_radial_bins(ra_center=244.26004*u.deg, dec_center=-22.97608*u.deg, nstars=100,
                            dlogr=0.3)

# The bin numbers form a new column in the table
print(input_data.data['bin'].max())
# 6
```

## Configuring the model parameters

MCMC-Dynamics comes with a number of models, each of which has its own set of parameters that
can be optimized to fit the data. The parameters are organized using a modified  version of
the [Parameters](mcmc_dynamics.parameter.Parameters) class from the
[lmfit](https://lmfit.github.io/lmfit-py/parameters.html) package.

Each model has a set of default parameter settings that are provided with the code. These can
be accessed as follows.

```python
# Get default parameter settings for ConstantFit model
from mcmc_dynamics.analysis import ConstantFit
parameters = ConstantFit.default_parameters()
parameters.pretty_print()
# Name           Value     Unit      Min      Max    Fixed Initials  Lnprior
# dec_center         0      deg      -90       90    False     None     None
# ra_center        180      deg        0      360    False     None     None
# sigma_max          0   km / s        0      inf    False rng.lognormal(size=n)     None
# v_maxx             0   km / s     -inf      inf    False rng.normal(size=n)     None
# v_maxy             0   km / s     -inf      inf    False rng.normal(size=n)     None
# v_sys              0   km / s     -inf      inf    False rng.normal(size=n)     None

# Fix systemic velocity to 20 km/s
from astropy import units as u
parameters['v_sys'].set(value=20.*u.km/u.s, fixed=True)
```

To generate initial values for the walkers during the MCMC analysis, each
[Parameters](mcmc_dynamics.parameter.Parameters) instance internally initializes a [numpy random
number generator](https://numpy.org/doc/stable/reference/random/index.html#random-sampling-numpy-random),
labelled `rng`.
To specify how the initials for an individual parameter should be determined, the `initials`
option of each parameter can be set as follows.
```python
# Draw initial values for the parameter v_maxx from a normal distribution with a mean of 0 km/s
# and a dispersion of 2 km/s.
parameters['v_maxx'].set(initials='rng.normal(loc=0, scale=2., size=n)')
```
Note that whenever accessing the random number generator, the `size=n` option must be
provided, so that the number of random samples will automatically match the number of walkers.

In a similar fashion, you can also set priors on individual parameters. For example, the following
will set a Gaussian prior on `v_maxx`, with a standard deviation of 5 km/s and centred around its
current `value`.
```python
parameters['v_maxx'].set(lnprior='-0.5*(val - v_maxx)**2/{std}**2 - 0.5*log(2.*pi*{std}**2)'.format(std=5))
parameters['v_maxx'].evaluate_lnprior(4)
# -2.548376445638773
```
Note that `val` always refers to the value for which the prior should be evaluated.

It is also possible to link different parameters. The following constrains `v_maxx` to a  narrow
range around the current value of `v_maxy`, essentially constraining the range of the position
angle of the rotation field.
```python
parameters['v_maxx'].set(lnprior='-0.5*(val - v_maxy)**2/{std}**2 - 0.5*log(2.*pi*{std}**2)'.format(std=0.5))
```

Finally, fixed limits for parameters (such as a lower limit of 0 km/s on `sigma_max` should be
specified using the `min` and `max` options, rather than `lnprior`.

## Fitting a model to a set of data

To run an MCMC analysis on a dataset prepared as specified under #preparing-the-data, execute the instance of the model
class and specify at the least the number of walkers and the number of steps that each walker should proceed. The
following illustrates how to fit a constant dispersion and rotation curve to the fourth bin of the M80 data preprocessed
as illustrated above.

```python
from mcmc_dynamics.analysis import ConstantFit
# initialize the analysis
cf = ConstantFit(data=input_data.fetch_radial_bin(4))

# change parameter settings
cf.parameters['ra_center'].set(value=244.26004, fixed=True)
cf.parameters['dec_center'].set(value=-22.97608, fixed=True)
cf.parameters['v_sys'].set(value=0, fixed=False, initials='rng.normal(loc=v_sys, scale=1, size=n)')
cf.parameters['sigma_max'].set(value=10, fixed=False, initials='rng.uniform(low=5, high=15, size=n)')
cf.parameters['v_maxx'].set(value=0., fixed=False, initials='rng.normal(size=n)')
cf.parameters['v_maxy'].set(value=0., fixed=False, initials='rng.normal(size=n)')

# run analysis
sampler = cf(n_walkers=20, n_steps=100)
# 100%|██████████| 100/100 [00:01<00:00, 66.14it/s]
```
Once the analysis has completed, you can obtain the best-fitting model parameters and their confidence intervals as
follows.
```python
# get best-fit parameters and their confidence intervals, discarding first half of chain as burn-in
# <QTable length=3>
# value         v_sys        ...        v_maxx              v_maxy      
#               km / s       ...        km / s              km / s      
#  str6        float64       ...       float64             float64      
# ------ ------------------- ... ------------------- -------------------
# median -0.8651516044297907 ...   1.911626821000814 -0.7407026554668175
#  uperr  0.4180807766182799 ...  0.4988910969529603  0.5986677410082195
#  loerr  0.3986672317227089 ... 0.49156382221041417   0.609144034225952
```
Furthermore, several methods are provided to visualize the behaviour of the walkers, create corner plots of the
a-posteriori distributions of the model parameters, or compute the radial profiles of dispersion and rotation
corresponding to the best-fitting model. For further information, consult the API documentation of the
[Runner](mcmc_dynamics.analysis.runner.Runner) class.
