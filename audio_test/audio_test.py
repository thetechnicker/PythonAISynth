from scr.fourier_neural_network import FourierNN
from scr import predefined_functions, utils
from context import scr
import multiprocessing
import os
import numpy as np
import torch
import sounddevice as sd
from audio_callback import audio_callback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _1():
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
    max_parralel_notes = 3
    data = list(zip(t, (predefined_functions.predefined_functions_dict'sin' for x in t)))

    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.train(test_data=t)

        t2 = np.array([2 * np.pi * frequencies[f] * np.linspace(0,
                      1, samplerate) for f in range(max_parralel_notes)])
        t2_tensor = torch.tensor(t2, dtype=torch.float32).to(fourier_nn.device)
        buffer_size = 512

        try:
            stream = sd.OutputStream(
                callback=lambda *args, **kwargs: utils.messure_time_taken(
                    'audio_callback', audio_callback, *args, **kwargs, wait=False),
                samplerate=samplerate, channels=1, blocksize=buffer_size
            )
            with stream:
                with open(os.devnull, "w") as f:
                    while True:
                        if hasattr(audio_callback, 'i'):
                            print(audio_callback.i * buffer_size, file=f)

        except KeyboardInterrupt:
            print(f"{(utils.messure_time_taken.time_taken) / 1_000_000_000}s")
            print("exit")
            exit()


if __name__ == "__main__":
    main()
