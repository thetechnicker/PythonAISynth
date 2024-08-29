import pygame
from scr.fourier_neural_network import FourierNN
from scr import utils
from pygame import mixer, midi, sndarray
import atexit
import multiprocessing
from multiprocessing import Process, Queue, current_process
import os
import sys
from tkinter import filedialog
import numpy as np
import pretty_midi


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
                synthesized_note = fourier_nn.synthesize(
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

    MIDI_NOTE_OF_FREQ_ONE = midi.frequency_to_midi(1)

    def __init__(self, fourier_nn, stdout: Queue, num_channels: int = 20):
        self.stdout = stdout
        self.pool = None
        self.live_synth: Process = None
        self.notes_ready = False
        self.fourier_nn: FourierNN = fourier_nn
        self.fs = 44100  # Sample rate
        self.num_channels = num_channels
        self.notes: dict = {}
        # pygame.init()
        mixer.init(frequency=44100, size=-16,
                   channels=2, buffer=1024)
        mixer.set_num_channels(self.num_channels)

        self.running_channels: dict[str, tuple[int, mixer.Channel]] = {}
        self.free_channel_ids = list(range(
            self.num_channels))
        self.generate_sounds()

    def generate_sound_wrapper(self, x, offset):
        midi_note, data = self.fourier_nn.synthesize_tuple(x)
        return midi_note+offset, data

    def generate_sounds(self):
        self.notes_ready = False

        timestep = 44100

        t = np.arange(0, 1, step=1 / timestep)

        data = self.fourier_nn.predict((2 * np.pi*t)-np.pi)

        peak_frequency = utils.calculate_peak_frequency(data)

        print("peak_frequency", peak_frequency)

        midi_offset = midi.frequency_to_midi(peak_frequency) \
            - self.MIDI_NOTE_OF_FREQ_ONE
        print("Midi Note offset:", midi_offset)
        self.pool = multiprocessing.Pool(processes=os.cpu_count())
        atexit.register(self.pool.terminate)
        atexit.register(self.pool.join)
        result_async = self.pool.starmap_async(self.generate_sound_wrapper,
                                               ((x-midi_offset, midi_offset) for x in range(128)))
        utils.run_after_ms(1000, self.monitor_note_generation, result_async)

    def monitor_note_generation(self, result_async):
        try:
            self.note_list = result_async.get(0.1)
        except multiprocessing.TimeoutError:
            utils.run_after_ms(
                1000, self.monitor_note_generation, result_async)
        else:
            print("sounds Generated")

            self.notes_ready = True
            self.play_init_sound()
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

    def live_synth_loop(self):
        midi.init()
        mixer.init(frequency=44100, size=-16,
                   channels=2, buffer=1024)
        mixer.set_num_channels(self.num_channels)
        for note, sound in self.note_list:
            stereo = np.repeat(sound.reshape(-1, 1), 2, axis=1)
            stereo_sound = pygame.sndarray.make_sound(stereo)
            self.notes[note] = stereo_sound

        input_id = midi.get_default_input_id()
        if input_id == -1:
            print("No MIDI input device found.")
            return

        midi_input = midi.Input(input_id)
        print("Live Synth is running")
        while True:
            if midi_input.poll():
                midi_events = midi_input.read(10)
                for midi_event, timestamp in midi_events:
                    if midi_event[0] == 144:
                        print("Note one")
                        try:
                            id = self.free_channel_ids.pop()
                            channel = mixer.Channel(id)
                            # channel.set_volume(0)
                            channel.play(
                                self.notes[midi_event[1]], fade_ms=500)
                            self.running_channels[midi_event[1]] = (
                                id, channel,)
                        except IndexError:
                            print("to many sounds playing")
                    elif midi_event[0] == 128:
                        print("Note off")
                        self.free_channel_ids.append(
                            self.running_channels[midi_event[1]][0])
                        self.running_channels[midi_event[1]][1].stop()

    def run_live_synth(self):
        if not self.notes_ready:
            return
        if not self.live_synth:
            self.live_synth = Process(target=self.live_synth_loop)
            self.live_synth.start()
        else:
            self.live_synth.terminate()
            self.live_synth.join()
        atexit.register(utils.DIE, self.live_synth)

    def play_sound(self, midi_note):
        id = self.free_channel_ids.pop()
        channel = mixer.Channel(id)
        channel.play(self.notes[midi_note], 200)

    def __getstate__(self) -> object:
        pool = self.pool
        live_synth = self.live_synth
        del self.pool
        del self.live_synth
        Synth_dict = self.__dict__.copy()
        self.pool = pool
        self.live_synth = live_synth
        return Synth_dict

    def __setstate__(self, state):
        # Load the model from a file after deserialization
        self.__dict__.update(state)
        if current_process().name != 'MainProcess':
            sys.stdout = utils.QueueSTD_OUT(queue=self.stdout)
