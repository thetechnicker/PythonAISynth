import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import torch


# autopep8: off
from context import scr
from scr import predefined_functions, utils
from scr.fourier_neural_network import FourierNN
# autopep8: on


def _1():
    # to avoid removing importent import of scr
    scr


def main():
    # mixed_precision.set_global_policy('mixed_float16')
    # tf.config.optimizer.set_jit(True)
    # Initialize variables
    samplerate = 44100
    # f = 1
    frequencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    t = np.linspace(0, 2*np.pi, 250)
    max_parralel_notes = 1
    max_parralel_notes = min(max(1, max_parralel_notes), len(frequencies))
    func_name = 'nice'
    data = list(
        zip(t, (predefined_functions.call_func(func_name, x) for x in t)))
    # print(data)
    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        # fourier_nn.load_new_model_from_file()
        fourier_nn.update_attribs(DEFAULT_FORIER_DEGREE=50)
        fourier_nn.train(test_data=t)

        # Initial setup
        fig, ax = plt.subplots()
        resolution = 1000
        x = np.linspace(-3*np.pi, 3*np.pi, resolution)
        ax.set_ylim(-5, 5)
        with torch.no_grad():
            y = fourier_nn.current_model(
                torch.tensor(
                    x,
                    dtype=torch.float32
                ).unsqueeze(1)
            )
            line, = ax.plot(x, y.numpy(), label='model')
            line1, = ax.plot(x, predefined_functions.call_func(
                func_name, x), label='actual data')

            # Function to update the plot
            def update_plot(event):
                # print(event)
                # print(ax.get_xlim(), ax.get_ylim())
                x_min, x_max = ax.get_xlim()
                x = np.linspace(x_min, x_max, resolution)
                y = fourier_nn.current_model(
                    torch.tensor(
                        x,
                        dtype=torch.float32
                    ).unsqueeze(1)
                )
                line.set_data(x, y.numpy())
                line1.set_data(x, predefined_functions.call_func(func_name, x))
                fig.canvas.draw_idle()

            # Connect the update function to the xlim and ylim changed events
            ax.callbacks.connect('xlim_changed', update_plot)
            ax.callbacks.connect('ylim_changed', update_plot)

            plt.show()


if __name__ == "__main__":
    main()
