from setuptools import setup, find_packages
from _version import version

setup(
    name='PythonAISynth',
    version=version,
    py_modules=['main', '_version'],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'numpy',
        'pygame',
        'scipy',
        'torch',
        'psutil',
        'numba',
        'pretty_midi',
        'sounddevice',
        'pyaudio',
        'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'PythonAISynth = main:main',
        ],
    },
)
