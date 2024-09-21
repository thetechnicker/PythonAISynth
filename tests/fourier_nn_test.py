import multiprocessing
import os
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import torch
import sounddevice as sd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
    frequencies = [220, 440, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200]
    t = np.linspace(-np.pi, np.pi, 250)
    max_parralel_notes = 5
    data = list(
        zip(t, (predefined_functions.predefined_functions_dict['sin'](x) for x in t)))
    # print(data)
    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.train(test_data=t)

        # utils.messure_time_taken(
        #     "predict", lambda x: [fourier_nn.predict(_x, 1) for _x in x], 2*np.pi * f * np.linspace(0, 1, timestep))
        t2 = np.array([2 * np.pi * frequencies[f] *
                       np.linspace(0, 1, samplerate).reshape(-1, 1) for f in range(max_parralel_notes)])
        t2_tensor = torch.tensor(t2, dtype=torch.float32).to(fourier_nn.device)
        print(t2_tensor.shape)
        buffer_size = 512
        a = np.zeros((max_parralel_notes, samplerate, 1))

        with torch.no_grad():
            for i in range((t2.shape[1]//buffer_size)+1):
                def generate():
                    for j in range(max_parralel_notes):
                        a[:, i*buffer_size:(i+1)*buffer_size] = fourier_nn.current_model(
                            t2_tensor[j, i*buffer_size:(i+1)*buffer_size]).cpu().numpy()
                utils.messure_time_taken("predict", generate, wait=False)

        # try:
        #     with torch.no_grad():
        #         i = 0
        #         while True:
        #             def generate():
        #                 y = []
        #                 for j in range(max_parralel_notes):
        #                     x = fourier_nn.current_model(
        #                         t2_tensor[j, i*buffer_size:(i+1)*buffer_size]).cpu().numpy()
        #                     # sd.play(x, blocking=False)
        #             utils.messure_time_taken("predict", generate, wait=False)
        #             i = (i+1) % (samplerate//buffer_size)

        # except KeyboardInterrupt:
        #     print("exit")
        #     exit()

        for i, (x, y) in enumerate(zip(t2, a)):
            plt.plot(x, y, label=f"{i}")

        plt.xlabel('x (radians)')
        plt.ylabel('sin(x)')
        plt.title('Sine Wave')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
