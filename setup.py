from distutils.core import setup

from setuptools import find_packages

setup(
    name='P34 ABM',
    description='Agent based model framework to simulate collectively foraging agents relying on their private and social'
                'visual cues. Written in pygame and python 3.7+',
    version='1.2.0',
    url='https://github.com/scioip34/ABM',
    maintainer='David Mezey and Dominik Deffner @ SCIoI',
    packages=find_packages(exclude=['tests']),
    package_data={'p34abm': ['*.txt']},
    python_requires=">=3.7",
    install_requires=[
        'pygame',
        # 'pygame-widgets',
        'numpy',
        # 'scipy',
        'matplotlib',
        'python-dotenv',
        # 'pandas',
        # 'influxdb<5.3.0',
        # 'opencv-python',
        # 'xvfbwrapper',
        'zarr',
        'torch',
        'cma'
    ],
    # extras_require={
    #     'test': [
    #         'bandit',
    #         'flake8',
    #         'pytest',
    #         'pytest-cov'
    #     ]
    # },
    entry_points={
        'console_scripts': [
            'abm=abm.start_sim:start',
            'EA=abm.start_EA:start_EA',
            'multi=abm.start_EA:start_EA_multirun',
        ]
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Operating System :: Other OS',
        'Programming Language :: Python :: 3.8'
    ],
    test_suite='tests',
    zip_safe=False
)
