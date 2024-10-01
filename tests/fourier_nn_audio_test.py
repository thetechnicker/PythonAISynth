import multiprocessing
import os
from threading import Thread
from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
import torch
import sounddevice as sd

# autopep8: off
from context import scr
from scr import predefined_functions, utils
from scr.fourier_neural_network import FourierNN
from scr.music import Synth2, Synth3
# autopep8: on


def _1():
    # to avoid removing importent import of scr
    scr


fourier_nn = None
t2_tensor = None
samplerate = 44100
frequencies = [220, 440, 660, 880, 1100, 1320, 1540, 1760, 1980, 2200]
t = np.linspace(-np.pi, np.pi, 250)
max_parralel_notes = 1
buffer_size = 265


def main():
    global fourier_nn, samplerate, frequencies, t, max_parralel_notes, t2_tensor, buffer_size
    data = list(
        zip(t, (predefined_functions.predefined_functions_dict['sin'](x) for x in t)))

    with multiprocessing.Manager() as manager:
        fourier_nn = FourierNN(lock=manager.Lock(), data=data)
        fourier_nn.train(test_data=t)

        # thread = Thread(target=stupid_thread, args=(fourier_nn,))
        # thread.start()
        # try:
        #     while thread.is_alive():
        #         pass
        # except KeyboardInterrupt:
        #     pass

        # stupid(fourier_nn)
        # stupid_thread(fourier_nn)

        # process = multiprocessing.Process(target=stupid, args=(fourier_nn,))
        # print("starting process")
        # process.start()
        # print("process has started")
        # good = False
        # try:
        #     while process.is_alive():
        #         pass
        # except KeyboardInterrupt:
        #     if process.is_alive():
        #         process.kill()
        # except:
        #     pass
        # print(process.exitcode)


def stupid_thread(fourier_nn):
    synth = Synth3(fourier_nn)
    # synth.current_notes.add(45)
    synth.run_live_synth()
    while True:
        if synth.live_synth:
            break
    try:
        while synth.live_synth.is_alive():
            pass
    except KeyboardInterrupt:
        if synth.live_synth.is_alive():
            synth.run_live_synth()


def audio_callback(outdata, frames, time, status):
    global fourier_nn, samplerate, frequencies, t, max_parralel_notes, t2_tensor, buffer_size
    if not hasattr(audio_callback, 'i'):
        audio_callback.i = 0
    i = audio_callback.i
    if status:
        print(status)

    a = torch.zeros((max_parralel_notes, frames))
    with torch.no_grad():
        for j in range(max_parralel_notes):
            a[j, :] = fourier_nn.current_model(
                utils.wrap_concat(t2_tensor[j], i, i+frames))

    y = torch.clamp(torch.sum(a, dim=0), min=-1, max=1).numpy()

    outdata[:] = y.reshape(-1, 1)
    audio_callback.i = (audio_callback.i + frames) % samplerate


def stupid(_fourier_nn):
    global fourier_nn, samplerate, frequencies, t, max_parralel_notes, buffer_size, t2_tensor
    fourier_nn = _fourier_nn
    fourier_nn.current_model.eval()
    t2 = np.array([2 * np.pi * frequencies[f] *
                   np.linspace(0, 1, samplerate) for f in range(max_parralel_notes)])
    t2_tensor = torch.tensor(t2, dtype=torch.float32)
    try:
        # i = 0
        stream = sd.OutputStream(
            callback=lambda *args, **kwargs: utils.messure_time_taken('audio_callback', audio_callback, *args, **kwargs, wait=False), samplerate=samplerate, channels=1, blocksize=buffer_size)
        with stream:
            with open(os.devnull, "w") as f:
                while True:
                    if hasattr(audio_callback, 'i'):
                        print(audio_callback.i*buffer_size, file=f)

    except KeyboardInterrupt:
        print(*utils.messure_time_taken.time_taken.items(), sep="\n")
        print("exit")
        exit()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
