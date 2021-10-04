from setuptools import setup

setup(
    name='Cestimii',
    url='https://github.com/uhlmanngroup/Cestimii',
    author='Simon Biberger',
    author_email='cestimii@biberger.xyz',
    packages=['cestimii'],
    install_requires=['numpy','scipy','matplotlib','napari','h5py','pillow'],
    version='0.6',
    license='BSD-3-Clause',
    description='Multiscale Curvature Estimation using Integral Invariants.',
    long_description=open('README.txt').read(),
    scripts = ['scripts/test.py']
)
