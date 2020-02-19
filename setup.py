from setuptools import setup, find_packages

setup(
    name='GANs and Variational Autoencoders',
    version='0.0.1',
    description='Materials for Course on GANs and Variational Autoencoders',
    license='BSD 3-clause license',
    maintainer='Andrei-Claudiu Roibu',
    maintainer_email='andrei-claudiu.roibu@dtc.ox.ac.uk',
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
        'pandas',
        'sklearn',
        'tensorflow',
        'theano',
        # 'autopep8',
        # 'pylint',
        'imageio',
        'natsort',
    ],
)