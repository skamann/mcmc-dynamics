from setuptools import setup

setup(
    name='mcmc-dynamics',
    version='0.2',
    packages=['example', 'mcmc_dynamics', 'mcmc_dynamics.utils', 'mcmc_dynamics.utils.files',
              'mcmc_dynamics.utils.plots', 'mcmc_dynamics.utils.science', 'mcmc_dynamics.utils.coordinates',
              'mcmc_dynamics.analysis', 'mcmc_dynamics.analysis.cjam', 'mcmc_dynamics.background'],
    url='',
    license='',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    install_requires=['numpy', 'astropy', 'scipy', 'matplotlib', 'corner', 'emcee', 'tqdm', 'pandas', 'pathos',
                      'asteval', 'lmfit'],
    description='Tools for maximum-likelihood analysis of radial velocity data'
)
