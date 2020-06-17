from setuptools import setup

setup(
    name='mcmc-dynamics',
    version='0.1',
    packages=['example', 'mcmc_dynamics', 'mcmc_dynamics.utils', 'mcmc_dynamics.utils.files',
              'mcmc_dynamics.utils.plots', 'mcmc_dynamics.utils.science', 'mcmc_dynamics.utils.coordinates',
              'mcmc_dynamics.analysis', 'mcmc_dynamics.analysis.cjam', 'mcmc_dynamics.background'],
    url='',
    license='',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    description='Tools for maximum-likelihood analysis of radial velocity data'
)
