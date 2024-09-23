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


def wrap_concat(tensor, idx1, idx2):
    length = tensor.size(0)

    if idx2 >= length:
        idx2 = idx2 % length

    if idx1 <= idx2:
        result = tensor[idx1:idx2]
    else:
        result = torch.cat((tensor[idx1:], tensor[:idx2]))

    return result


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

            a = torch.zeros((max_parralel_notes, frames, 1),
                            device=fourier_nn.device)
            with torch.no_grad():
                for j in range(max_parralel_notes):
                    # t_ = (i+j*10)
                    a[j, :] = fourier_nn.current_model(
                        wrap_concat(t2_tensor[j], i, i+frames))
            y = torch.clamp(torch.sum(a, dim=0), min=-1, max=1).cpu().numpy()

            outdata[:] = y.reshape(-1, 1)
            audio_callback.i = (audio_callback.i + frames) % samplerate

        try:
            # i = 0
            stream = sd.OutputStream(
                callback=lambda *args, **kwargs: utils.messure_time_taken('audio_callback', audio_callback, *args, **kwargs, wait=False), samplerate=samplerate, channels=1, blocksize=buffer_size)
            with stream:
                with open(os.devnull, "w") as f:
                    while True:
                        if hasattr(audio_callback, 'i'):
                            # print("", end="")
                            print(audio_callback.i*buffer_size, file=f)
                        # pass

        except KeyboardInterrupt:
            print(*utils.messure_time_taken.time_taken.items(), sep="\n")
            print("exit")
            exit()


if __name__ == "__main__":
    main()
