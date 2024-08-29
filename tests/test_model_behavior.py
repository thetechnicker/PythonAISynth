from scipy.fft import fft, fftfreq
import multiprocessing
import numpy as np
from context import scr
from scr.fourier_neural_network import FourierNN
import matplotlib.pyplot as plt

from scr.utils import calculate_peak_frequency


def main():
    step = 44100
    time = 1
    lst = 2 * np.pi * np.linspace(0, 1, step)

    with multiprocessing.Manager() as manager:
        lock = manager.Lock()
        fourier_nn = FourierNN(lock)
        fourier_nn.load_new_model_from_file(
            "tmp/nice.keras")  # replace with actual net
        data = fourier_nn.predict(lst)
    peak_frequency = calculate_peak_frequency(np.sin(lst))
    print(f"The peak frequency is {peak_frequency} Hz")
    plt.plot(lst, data)

    # Display the plot
    plt.show()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
