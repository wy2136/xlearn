"""Build and install the xlearn package"""

from setuptools import setup


setup(
    name='xlearn',
    version='0.1.0',
    description='xarray-aware scikit-learn',
    url='https://github.com/wy2136/xlearn',
    author='Wenchang Yang',
    author_email='yang.wenchang@uci.edu',
    long_description='''
        Extend scikit-learn to accept xarray DataArray as input data to fit or
        predict''',
    packages = ['xlearn'],
    install_requires=['numpy', 'matplotlib', 'xarray', 'scikit-learn']
)
