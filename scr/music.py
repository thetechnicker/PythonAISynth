import time
from scr.fourier_neural_network import FourierNN
from scr import utils
import mido
from pygame import mixer, midi, sndarray
import atexit
import multiprocessing
from multiprocessing import Process, Queue, current_process
import os
import sys
from threading import Thread
from tkinter import filedialog
import numpy as np
import pretty_midi
from scipy.fftpack import fft


def musik_from_file(fourier_nn: FourierNN):
    import sounddevice as sd
    midi_file = filedialog.askopenfilename()
    if not midi_file:
        return
    print(midi_file)
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    # print(dir(midi_data))
    notes_list = []
    # Iterate over all instruments and notes in the MIDI file
    if fourier_nn:
        for instrument in midi_data.instruments:
            # print(dir(instrument))
            for note_a, note_b in utils.note_iterator(instrument.notes):
                # node_a
                print(note_a)
                duration = note_a.end - note_a.start
                synthesized_note = fourier_nn.synthesize_2(
                    midi_note=note_a.pitch-12, duration=duration, sample_rate=44100)
                print(synthesized_note.shape)
                notes_list.append(synthesized_note)

                if note_b:
                    # pause
                    print("Pause")
                    duration = abs(note_b.start - note_a.end)
                    synthesized_note = np.zeros((int(duration*44100), 1))
                    print(synthesized_note.shape)
                    notes_list.append(synthesized_note)

                # # node_b
                # print(note_b)
                # duration = note_b.end - note_b.start
                # synthesized_note = fourier_nn.synthesize_2(
                #     midi_note=note_b.pitch, duration=duration, sample_rate=44100)
                # print(synthesized_note.shape)
                # notes_list.append(synthesized_note)
            break
    output = np.concatenate(notes_list)
    sd.play(output, 44100)


class Synth():
    def __init__(self, master, fourier_nn, stdout: Queue, num_channels: int = 20):
        self.master = master
        self.stdout = stdout
        self.fourier_nn: FourierNN = fourier_nn
        self.fs = 44100  # Sample rate
        self.num_channels = num_channels
        # pygame.init()
        mixer.init(frequency=44100, size=-16,
                   channels=2, buffer=1024)
        mixer.set_num_channels(self.num_channels)

        self.running_channels: dict[str, tuple[int, mixer.Channel]] = {}
        self.free_channel_ids = list(range(
            self.num_channels))
        self.generate_sounds()

    def generate_sound_wrapper(self, midi_note):
        return self.fourier_nn.synthesize_3()

    def generate_sounds(self):
        t = np.arange(0, 1, 1/44100)
        y = self.fourier_nn.predict(2 * np.pi*t)
        fft_vals = fft(y)

        # Get absolute value of FFT values (to get magnitude)
        fft_abs = np.abs(fft_vals)

        # Get frequency values for each FFT bin
        # The second argument is the sample spacing (inverse of the sample rate)
        freqs = np.fft.fftfreq(len(y), 1/44100)

        # Find the frequency where the magnitude of FFT is maximum
        dominant_freq = freqs[np.argmax(fft_abs)]
        print(dominant_freq)
        midi_offset = midi.frequency_to_midi(dominant_freq)

        self.pool = multiprocessing.Pool(processes=os.cpu_count())
        atexit.register(self.pool.terminate)
        atexit.register(self.pool.join)
        self.result_async = self.pool.map_async(self.generate_sound_wrapper,
                                                (x-midi_offset for x in range(128)))
        self.master.after(1000, self.monitor_note_generation)

    def monitor_note_generation(self):
        try:
            self.notes = self.result_async.get(0.1)
        except multiprocessing.TimeoutError:
            self.master.after(1000, self.monitor_note_generation)
        else:
            self.pool.close()
            self.pool.join()
            atexit.unregister(self.pool.terminate)
            atexit.unregister(self.pool.join)

    def play_init_sound(self):
        f1 = 440  # Frequency of the "duuu" sound (in Hz)
        f2 = 880  # Frequency of the "dib" sound (in Hz)
        t1 = 0.8  # Duration of the "duuu" sound (in seconds)
        t2 = 0.2  # Duration of the "dib" sound (in seconds)
        t = np.arange(int(t1 * self.fs)) / self.fs
        sound1 = 0.5 * np.sin(2 * np.pi * f1 * t)

        # Generate the "dib" sound
        t = np.arange(int(t2 * self.fs)) / self.fs
        sound2 = 0.5 * np.sin(2 * np.pi * f2 * t)

        # Concatenate the two sounds
        audio = np.concatenate([sound1, sound2])
        output = np.array(
            audio * 32767 / np.max(np.abs(audio)) / 2).astype(np.int16)
        stereo_sine_wave = np.repeat(output.reshape(-1, 1), 2, axis=1)
        sound = sndarray.make_sound(stereo_sine_wave)

        channel = mixer.Channel(0)
        channel.play(sound)
        while (channel.get_busy()):
            pass
        print("Ready")

    def __setstate__(self, state):
        # Load the model from a file after deserialization
        self.__dict__.update(state)
        if current_process().name != 'MainProcess':
            sys.stdout = utils.QueueSTD_OUT(self.stdout)


# try:
#     import pygame
#     import mido
#     port_name = 'AI_SYNTH'
#     virtual = True
#     if not os.getenv('HAS_RUN_INIT'):
#         try:
#             mido.open_input(port_name, virtual=virtual).close()
#         except NotImplementedError:
#             print("test loopBe midi")
#             port_name = "LoopBe Internal MIDI 0"
#             virtual = False
#             mido.open_input(port_name, virtual=virtual).close()
#             print("loopbe works")
#         print("live music possible")
#     proc = None

#     def midi_proc(note_list, port_name: str, virtual: bool, stdout: Queue):

#         if stdout:
#             sys.stdout = utils.QueueSTD_OUT(stdout)
#         print("start Midi")
#         pygame.init()
#         pygame.mixer.init(frequency=44100, size=-16,
#                           channels=2, buffer=1024)
#         notes: dict = {}

#         for note, sound in note_list:
#             stereo = np.repeat(sound.reshape(-1, 1), 2, axis=1)
#             stereo_sound = pygame.sndarray.make_sound(stereo)
#             notes[note] = stereo_sound

#         print(notes.keys())

#         # Set the number of channels to 20
#         pygame.mixer.set_num_channels(20)

#         fs = 44100  # Sample rate
#         f1 = 440  # Frequency of the "duuu" sound (in Hz)
#         f2 = 880  # Frequency of the "dib" sound (in Hz)
#         t1 = 0.8  # Duration of the "duuu" sound (in seconds)
#         t2 = 0.2  # Duration of the "dib" sound (in seconds)

#         # Generate the "duuu" sound
#         t = np.arange(int(t1 * fs)) / fs
#         sound1 = 0.5 * np.sin(2 * np.pi * f1 * t)

#         # Generate the "dib" sound
#         t = np.arange(int(t2 * fs)) / fs
#         sound2 = 0.5 * np.sin(2 * np.pi * f2 * t)

#         # Concatenate the two sounds
#         audio = np.concatenate([sound1, sound2])
#         output = np.array(
#             audio * 32767 / np.max(np.abs(audio)) / 2).astype(np.int16)
#         stereo_sine_wave = np.repeat(output.reshape(-1, 1), 2, axis=1)
#         sound = pygame.sndarray.make_sound(stereo_sine_wave)

#         # Play the sound on a specific channel
#         running_channels: dict[str, tuple[int, pygame.mixer.Channel]] = {}
#         free_channel_ids = list(range(20))
#         channel = pygame.mixer.Channel(0)
#         channel.play(sound)
#         while (channel.get_busy()):
#             pass
#         print("Ready")

#         with mido.open_input(port_name, virtual=virtual) as midiin:
#             while True:
#                 for msg in midiin.iter_pending():
#                     print(msg)
#                     if msg.type == 'note_off':
#                         free_channel_ids.append(
#                             running_channels[msg.note][0])
#                         running_channels[msg.note][1].stop()
#                     elif msg.type == 'note_on':
#                         try:
#                             id = free_channel_ids.pop()
#                             channel = pygame.mixer.Channel(id)
#                             channel.set_volume(0)
#                             channel.play(notes[msg.note], fade_ms=500)
#                             running_channels[msg.note] = (id, channel,)
#                         except IndexError:
#                             print("to many sounds playing")

#     def start_midi_process(fourier_nn: FourierNN, stdout: Queue):
#         print("test")
#         global proc
#         global port_name
#         global virtual
#         t = np.arange(0, 1, 1/44100)
#         y = fourier_nn.predict(2 * np.pi*t)
#         fft_vals = fft(y)

#         # Get absolute value of FFT values (to get magnitude)
#         fft_abs = np.abs(fft_vals)

#         # Get frequency values for each FFT bin
#         # The second argument is the sample spacing (inverse of the sample rate)
#         freqs = np.fft.fftfreq(len(y), 1/44100)

#         # Find the frequency where the magnitude of FFT is maximum
#         dominant_freq = freqs[np.argmax(fft_abs)]

#         print("dominant_freq:", dominant_freq)
#         fourier_nn.save_tmp_model()
#         with multiprocessing.Pool(processes=os.cpu_count()) as pool:
#             try:
#                 results = pool.map(
#                     fourier_nn.synthesize_3, range(128))
#             except Exception as e:
#                 print("error", e)
#                 pool.terminate()
#                 pool.join()
#                 return
#         proc = Process(target=midi_proc, args=(
#             results, port_name, virtual, stdout,))
#         proc.start()

#     def midi_to_musik_live(fourier_nn: FourierNN, stdout: Queue):
#         os.environ['play'] = 'true'
#         global proc
#         if proc:
#             utils.DIE(proc)
#             proc = None

#         fourier_nn.save_tmp_model()

#         t = Thread(target=start_midi_process, args=(fourier_nn, stdout, ))
#         # t.daemon = True
#         t.start()
#         atexit.register(t.join)
#         atexit.register(utils.DIE, proc)

# except Exception as e:
#     print("live music NOT possible", e, sep="\n")
