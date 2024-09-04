import multiprocessing
import os
from matplotlib import ticker
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.fft import dst
import tensorflow as tf
from tensorflow.keras import mixed_precision

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# autopep8: off
from context import scr
from scr import predefined_functions
from scr.fourier_neural_network import FourierNN
from scr import utils
from scr.music import Synth
# autopep8: on


def main():
    mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(True)
    # Initialize variables
    timestep = 44100
    f = 440

    t = np.linspace(-np.pi, np.pi, 250)
    data = list(
        zip(t, (predefined_functions.predefined_functions_dict['sin'](x) for x in t)))
    # print(data)
    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.train(test_data=t)

        # utils.messure_time_taken(
        #     "predict", lambda x: [fourier_nn.predict(_x, 1) for _x in x], 2*np.pi * f * np.linspace(0, 1, timestep))
        x = 2*np.pi * np.linspace(0, 1, timestep)
        buffer_size = 512
        a = np.zeros((timestep, 1))
        for i in range((len(x)//buffer_size)+1):
            a[i*buffer_size:(i+1)*buffer_size] = utils.messure_time_taken(
                "predict", fourier_nn.predict, x[i*buffer_size:(i+1)*buffer_size], wait=False)

        fig, ax = plt.subplots()
        ax.plot(x, a)

        # Set the x-axis major locator to multiples of π
        ax.xaxis.set_major_locator(ticker.MultipleLocator(base=np.pi))

        # Format the x-axis labels to show multiples of π
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda val, pos: '{:.0g}π'.format(val / np.pi) if val != 0 else '0'))

        plt.xlabel('x (radians)')
        plt.ylabel('sin(x)')
        plt.title('Sine Wave')
        plt.grid(True)
        plt.show()
        # b = np.array([predefined_functions.predefined_functions_dict['sin'](_x)
        #              for _x in x])
        # print(np.allclose(a, b, atol=0.5))


if __name__ == "__main__":
    main()
