import unittest
import numpy as np
from context import src
from src.utils import calculate_peak_frequency


class TestCalculatePeakFrequency(unittest.TestCase):
    def test_peak_frequency(self):
        timestep = 44100
        t = np.arange(0, 1, step=1 / timestep)
        f = 440
        # Generate the original signal
        data = np.sin(2 * np.pi * f * t) - np.pi
        peak_freq = calculate_peak_frequency(data)

        self.assertEqual(
            peak_freq, f, "The peak frequency should be equal to the generated frequency.")


if __name__ == '__main__':
    unittest.main()
