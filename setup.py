from setuptools import find_packages, setup

setup(
    name='mcmc-dynamics',
    version='0.2',
    packages=find_packages("./"),
    scripts=['bin/cluster_elongation.py'],
    package_data={"": ["*.json"]},
    url='',
    license='',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    install_requires=['numpy>=1.17', 'astropy', 'scipy', 'matplotlib', 'corner', 'emcee', 'tqdm', 'pandas', 'pathos',
                      'asteval', 'lmfit'],
    description='Tools for maximum-likelihood analysis of radial velocity data'
)
