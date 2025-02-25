from setuptools import setup, find_packages
from pythonaisynth._version import version

setup(
    name='PythonAISynth',
    version=version,
    packages=find_packages('.', ['tests', 'audio_stuff']),
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
            'PythonAISynth = pythonaisynth.main:main',
        ],
    },
)
