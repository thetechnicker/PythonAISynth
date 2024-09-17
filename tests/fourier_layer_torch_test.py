import os
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# autopep8: off
from context import scr
from scr import predefined_functions, utils
from scr.fourier_neural_network import FourierLayer
# autopep8: on


def main():
    # Set the Fourier degree
    fourier_degree = 250

    # Initialize the FourierLayer with the specified degree
    fourier_layer = FourierLayer(fourier_degree)

    # Generate input data
    x = torch.linspace(0, 2 * np.pi, 44100).unsqueeze(1)

    # Apply the FourierLayer
    output = utils.messure_time_taken("fourier_layer", fourier_layer, x)

    # Extract sine and cosine components
    sine_components = output[:, :fourier_degree].detach().numpy()
    cosine_components = output[:, fourier_degree:].detach().numpy()

    plt.figure(figsize=(15, 6))

    for i in range(5):
        plt.subplot(2, 5, i+1)
        plt.plot(x.numpy(), sine_components[:, i*(fourier_degree//5)])
        plt.title(f'Sine {i*(fourier_degree//5)}')
        plt.subplot(2, 5, i+6)
        plt.plot(x.numpy(), cosine_components[:, i*(fourier_degree//5)])
        plt.title(f'Cosine {i*(fourier_degree//5)}')

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
