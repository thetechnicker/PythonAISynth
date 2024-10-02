from setuptools import setup, find_packages
from _version import version

setup(
    name='PythonAISynth',
    version=version,
    py_modules=['main', '_version'],
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch',
        'sounddevice',
        'pretty_midi',
        'scipy',
        'matplotlib',
        'sympy',
        'pygame',
        'python-rtmidi',
    ],
    entry_points={
        'console_scripts': [
            'PythonAISynth = main:main',
        ],
    },
)
