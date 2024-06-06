import numpy as np


def midi_to_freq(midi_note):
    return 440.0 * np.power(2.0, (midi_note - 69) / 12.0)