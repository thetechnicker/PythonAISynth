import multiprocessing
import os
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# autopep8: off
from context import scr
from scr import predefined_functions
from scr.fourier_neural_network import FourierNN
# autopep8: on


def _1():
    # to avoid removing importent import of scr
    scr


def main():
    # mixed_precision.set_global_policy('mixed_float16')
    # tf.config.optimizer.set_jit(True)
    # Initialize variables
    timestep = 44100
    f = 1
    t = np.linspace(-np.pi, np.pi, 250)
    data = list(
        zip(t, (predefined_functions.predefined_functions_dict['sin'](x) for x in t)))
    # print(data)
    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.train(test_data=t)

        # utils.messure_time_taken(
        #     "predict", lambda x: [fourier_nn.predict(_x, 1) for _x in x], 2*np.pi * f * np.linspace(0, 1, timestep))
        t2 = 2 * np.pi * f * np.linspace(0, 1, timestep)
        # buffer_size = 1024  # 512
        # a = np.zeros((timestep, 1))
        # x = FourierNN.fourier_basis_numba(
        #     t2, FourierNN.precompute_indices(fourier_nn.fourier_degree))

        # for i in range((len(x)//buffer_size)+1):
        #     a[i*buffer_size:(i+1)*buffer_size] = utils.messure_time_taken('predict',
        #                                                                   fourier_nn.current_model.predict,
        #                                                                   x[i*buffer_size:(
        #                                                                       i+1)*buffer_size],
        #                                                                   batch_size=buffer_size,
        #                                                                   wait=False)
        # a[i*buffer_size:(i+1)*buffer_size] = utils.messure_time_taken(
        #     "predict", fourier_nn.predict, x[i*buffer_size:(i+1)*buffer_size], wait=False)
        with torch.no_grad():
            a = fourier_nn.current_model(torch.tensor(
                t2, dtype=torch.float32).to(fourier_nn.device)).cpu().numpy()
        # try:
        #     a = fourier_nn.current_model.predict(
        #         x,
        #         batch_size=len(x))
        #     # verbose=0)
        #     # max_i = (len(x)//buffer_size)+1
        #     # i = 0
        #     while True:
        #         sd.wait()
        #         sd.play(a)
        #         # i = (i+1) % (max_i)
        # except KeyboardInterrupt:
        #     print("exit")
        #     exit()

        fig, ax = plt.subplots()
        ax.plot(t2, a)

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
