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

    # Determine the number of rows and columns
    total_functions = (fourier_degree // 5 + 1) * 2
    cols = max(5, total_functions // 2)
    rows = (total_functions + cols - 1) // cols
    if rows % 2 != 0:
        rows += 1

    # Plot every fourier_degree // 5 sine and cosine function
    plt.figure(figsize=(12, 5))

    for i in range(0, fourier_degree, fourier_degree//5):
        plt.subplot(rows, cols, i//(fourier_degree//5) + 1)
        plt.plot(x.numpy(), sine_components[:, i])
        plt.title(f'Sine {i+1}')
        plt.subplot(rows, cols, i//(fourier_degree//5) +
                    total_functions//2 + 1)
        plt.plot(x.numpy(), cosine_components[:, i])
        plt.title(f'Cosine {i+1}')

    # plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
