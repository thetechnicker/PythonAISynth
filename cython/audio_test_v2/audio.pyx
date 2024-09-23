# cython: language_level=3
import multiprocessing
import os
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import torch
import sounddevice as sd
from context import scr
from scr import predefined_functions, utils
from scr.fourier_neural_network import FourierNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def _1():
    # to avoid removing importent import of scr
    scr

cdef wrap_concat(tensor, idx1, idx2):
    cdef int length = tensor.size(0)

    if idx2 >= length:
        idx2 = idx2 % length

    if idx1 <= idx2:
        result = tensor[idx1:idx2]
    else:
        result = torch.cat((tensor[idx1:], tensor[:idx2]))

    return result

cdef main():
    cdef int samplerate = 44100
    cdef list frequencies = [220, 440, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200]
    t = np.linspace(-np.pi, np.pi, 250)
    cdef int max_parralel_notes = 1
    cdef list data = list(
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

cpdef run():
    main()

if __name__ == "__main__":
    main()
