import setuptools

tests_require = [
    'pytest>=6.1.1',
    'coverage>=5.3',
    'pytest-cov>=2.10.1'
]

setuptools.setup(
    name='streaky-stability',
    version='0.0.1',
    description='Routines to calculate the linear stability of a streaky flow',
    packages=setuptools.find_packages(),
    install_requires=[
        'scipy==1.6.3',
    ],
    tests_require=tests_require,
    extras_requires={'test': tests_require}
)
