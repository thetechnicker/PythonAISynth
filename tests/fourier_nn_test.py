import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

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
    frequencies = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    t = np.linspace(-np.pi, np.pi, 250)
    max_parralel_notes = 1
    max_parralel_notes = min(max(1, max_parralel_notes), len(frequencies))
    data = list(
        zip(t, (predefined_functions.predefined_functions_dict['extreme'](x) for x in t)))
    # print(data)
    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.update_attribs(fourier_degree=100)
        fourier_nn.train(test_data=t)

        # utils.messure_time_taken(
        #     "predict", lambda x: [fourier_nn.predict(_x, 1) for _x in x], 2*np.pi * f * np.linspace(0, 1, timestep))
        t2 = np.array([2 * np.pi * frequencies[f] *
                       np.linspace(0, 1, samplerate) for f in range(max_parralel_notes)])
        t2_tensor = torch.tensor(t2, dtype=torch.float32).to(fourier_nn.device)
        print(t2_tensor.shape)
        buffer_size = 512
        a = np.zeros((max_parralel_notes, samplerate, 1))

        with torch.no_grad():
            for i in range((t2.shape[1]//buffer_size)+1):
                def generate():
                    for j in range(max_parralel_notes):
                        a[j, i*buffer_size:(i+1)*buffer_size] = fourier_nn.current_model(
                            t2_tensor[j, i*buffer_size:(i+1)*buffer_size]).cpu().numpy()
                utils.messure_time_taken("predict", generate, wait=False)

        print(*utils.messure_time_taken.time_taken.items(), sep="\n")
        for name, param in fourier_nn.current_model.named_parameters():
            print(
                f"Layer: {name} | Size: {param.size()} | Values:\n{param[:2]}\n------------------------------")

        x = 2*np.pi*np.linspace(0, 1, samplerate)
        for i, (y) in enumerate(a):
            plt.plot(x, y, label=f"{i}")

        plt.xlabel('x (radians)')
        plt.ylabel('sin(x)')
        plt.title('Sine Wave')
        plt.grid(True)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
