# with torch.no_grad():
#     for i in range((t2.shape[1]//buffer_size)+1):
#         def generate():
#             for j in range(max_parralel_notes):
#                 a[j, i*buffer_size:(i+1)*buffer_size] = fourier_nn.current_model(
#                     t2_tensor[j, i*buffer_size:(i+1)*buffer_size]).cpu().numpy()
#         utils.messure_time_taken("predict", generate, wait=False)
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
    samplerate = 44100
    frequencies = [220, 440, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200]
    t = np.linspace(-np.pi, np.pi, 250)
    max_parralel_notes = 1
    data = list(
        zip(t, (predefined_functions.predefined_functions_dict['sin'](x) for x in t)))

    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.train(test_data=t)

        t2 = np.array([2 * np.pi * frequencies[f] *
                       np.linspace(0, 1, samplerate) for f in range(max_parralel_notes)])
        t2_tensor = torch.tensor(t2, dtype=torch.float32).to(fourier_nn.device)
        buffer_size = 512

        def audio_callback(outdata, frames, time, status):
            if not hasattr(audio_callback, 'i'):
                audio_callback.i = 0
            i = audio_callback.i
            if status:
                print(status)

            a = np.zeros((frames, 1))
            with torch.no_grad():
                # for j in range(max_parralel_notes):
                a[:] = fourier_nn.current_model(
                    t2_tensor[0, i*frames:(i+1)*frames]).cpu().numpy()
            # y = np.sum(a, 0)
            y = a
            # Apply a simple gain effect
            print(a)
            gain = 0.5
            y = y * gain
            outdata[:] = y.reshape(-1, 1)
            audio_callback.i = (i+1) % (samplerate//buffer_size)

        try:
            # i = 0
            stream = sd.OutputStream(
                callback=audio_callback, samplerate=samplerate, channels=1, blocksize=buffer_size)
            with stream:
                while True:
                    # print(i*buffer_size)
                    # i = (i+1) % (samplerate//buffer_size)
                    pass

        except KeyboardInterrupt:
            print("exit")
            exit()


if __name__ == "__main__":
    main()
