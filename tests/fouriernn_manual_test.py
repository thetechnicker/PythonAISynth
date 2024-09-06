import numpy as np
# import tensorflow as tf
import keras
# import matplotlib
# matplotlib.use('TkAgg')
from context import scr
from scr.fourier_neural_network import FourierNN


if __name__ == '__main__':
    # Test the class with custom data points
    data = [(x, np.sin(x * keras.activations.relu(x)))
            for x in np.linspace(np.pi, -np.pi, 100)]
    fourier_nn = FourierNN(data)
    fourier_nn.train_and_plot()
