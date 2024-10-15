import mido
import scipy
import torch
from src.fourier_neural_network import FourierNN
from src import utils
import atexit
from multiprocessing import Process, Queue, current_process
import sys
from tkinter import filedialog
import numpy as np
import pretty_midi
import sounddevice as sd
import pyaudio


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
            note_list = []
            for note_a, note_b in utils.note_iterator(instrument.notes):
                # node_a
                print(note_a)
                duration = note_a.end - note_a.start
                synthesized_note = fourier_nn.synthesize(
                    midi_note=note_a.pitch-12, duration=duration, sample_rate=44100)
                print(synthesized_note.shape)
                note_list.append(synthesized_note)

                if note_b:
                    # pause
                    print("Pause")
                    duration = abs(note_b.start - note_a.end)
                    synthesized_note = np.zeros((int(duration*44100), 1))
                    print(synthesized_note.shape)
                    note_list.append(synthesized_note)

                # # node_b
                # print(note_b)
                # duration = note_b.end - note_b.start
                # synthesized_note = fourier_nn.synthesize_2(
                #     midi_note=note_b.pitch, duration=duration, sample_rate=44100)
                # print(synthesized_note.shape)
                # notes_list.append(synthesized_note)
            notes_list.append(np.concatenate(note_list))

    output = np.sum(notes_list)
    output = rescale_audio(output)
    sd.play(output, 44100)


def rescale_audio(audio):
    max_val = max(abs(audio.min()), audio.max())
    if max_val == 0:
        return audio
    return audio / max_val


def apply_reverb(audio, decay=0.5, delay=0.02, fs=44100):
    delay_samples = int(delay * fs)
    impulse_response = np.zeros((delay_samples, 2))
    impulse_response[0, :] = 1
    impulse_response[-1, :] = decay
    reverb_audio = np.zeros_like(audio)
    for channel in range(audio.shape[1]):
        reverb_audio[:, channel] = scipy.signal.fftconvolve(
            audio[:, channel], impulse_response[:, channel])[:len(audio)]
    return reverb_audio


def apply_echo(audio, delay=0.2, decay=0.5, fs=44100):
    delay_samples = int(delay * fs)
    echo_audio = np.zeros((len(audio) + delay_samples))
    echo_audio[:len(audio)] = audio
    echo_audio[delay_samples:] += decay * audio
    echo_audio = echo_audio[:len(audio)]
    return echo_audio


def apply_chorus(audio, rate=1.5, depth=0.02, fs=44100):
    t = np.arange(len(audio)) / fs
    mod = depth * np.sin(2 * np.pi * rate * t)
    chorus_audio = np.zeros_like(audio)
    for channel in range(audio.shape[1]):
        for i in range(len(audio)):
            delay = int(mod[i] * fs)
            if i + delay < len(audio):
                chorus_audio[i, channel] = audio[i + delay, channel]
            else:
                chorus_audio[i, channel] = audio[i, channel]
    return chorus_audio


def apply_distortion(audio, gain=2.0, threshold=0.5):
    distorted_audio = gain * audio
    distorted_audio[distorted_audio > threshold] = threshold
    distorted_audio[distorted_audio < -threshold] = -threshold
    return distorted_audio


class ADSR:
    def __init__(self, attack_time, decay_time, sustain_level, release_time, sample_rate):
        self.attack_samples = int(attack_time * sample_rate)
        self.decay_samples = int(decay_time * sample_rate)
        self.ad_samples = self.attack_samples + self.decay_samples
        self.sustain_level = sustain_level
        self.release_samples = int(release_time * sample_rate)

        self.ads_envelope = np.concatenate((
            np.linspace(0, 1, self.attack_samples),
            np.linspace(1, self.sustain_level, self.decay_samples)
        ))
        self.r_envelope = np.linspace(
            self.sustain_level, 0, self.release_samples)

        self._r_counter = [0 for _ in range(128)]
        self._ads_counter = [0 for _ in range(128)]

    def reset_note(self, note_num):
        self._ads_counter[note_num] = 0
        self._r_counter[note_num] = 0

    def has_note_ended(self, note_num) -> bool:
        return self._r_counter[note_num] >= self.release_samples

    def get_ads_envelope(self, note_num, frame):
        start = self._ads_counter[note_num]
        end = start + frame
        self._ads_counter[note_num] += frame

        if start > self.ad_samples:
            return np.ones(frame) * self.sustain_level

        envelope = np.zeros(frame)
        if end > self.ad_samples:
            envelope[:self.ad_samples -
                     start] = self.ads_envelope[start:self.ad_samples]
            envelope[self.ad_samples -
                     start:] = np.ones(end - self.ad_samples)*self.sustain_level
        else:
            envelope[:] = self.ads_envelope[start:end]

        return envelope

    def get_r_envelope(self, note_num, frame):
        start = self._r_counter[note_num]
        end = start + frame
        self._r_counter[note_num] += frame

        if start > self.release_samples:
            return np.zeros(frame)

        envelope = np.zeros(frame)
        if end <= self.release_samples:
            envelope[:] = self.r_envelope[start:end]
        else:
            envelope[:self.release_samples -
                     start] = self.r_envelope[start:self.release_samples]
            # After release, the envelope is 0
            envelope[self.release_samples - start:] = 0
        return envelope


class Echo_torch:
    def __init__(self, sample_size, delay_time_ms, feedback, wet_dry_mix, device='cpu'):
        self.sample_size = sample_size
        # Convert ms to samples (assuming 44.1 kHz sample rate)
        self.delay_time = int(delay_time_ms * 44.1)
        self.feedback = feedback
        self.wet_dry_mix = wet_dry_mix
        self.device = device

        # Initialize the past buffer to store delayed samples
        self.past = torch.zeros(
            (self.delay_time + 1, sample_size), device=device)

    def apply(self, sound):
        # Ensure sound is a tensor of shape (num_samples, sample_size)
        if sound.dim() != 2 or sound.size(1) != self.sample_size:
            raise ValueError(
                "Input sound must be a 2D tensor with shape (num_samples, sample_size)")

        # Create an output tensor
        output = torch.zeros_like(sound, device=self.device)

        for i in range(sound.size(0)):
            # Get the current sample
            current_sample = sound[i]

            # Get the delayed sample from the past buffer
            delayed_sample = self.past[-self.delay_time] if i >= self.delay_time else torch.zeros(
                self.sample_size, device=self.device)

            # Calculate the echo effect
            echo_sample = (current_sample + delayed_sample *
                           self.feedback) * self.wet_dry_mix

            # Store the current sample in the past buffer
            self.past = torch.roll(self.past, shifts=-1, dims=0)
            self.past[-1] = current_sample

            # Combine the original sound and the echo
            output[i] = current_sample * (1 - self.wet_dry_mix) + echo_sample

        return output


class Echo:
    def __init__(self, sample_size, delay_time_ms, feedback, wet_dry_mix):
        self.sample_size = sample_size
        # Convert ms to samples (assuming 44.1 kHz sample rate)
        # Convert ms to seconds
        self.delay_time = int(delay_time_ms * 44.1 / 1000)
        self.feedback = feedback
        self.wet_dry_mix = wet_dry_mix

        # Initialize the past buffer to store delayed samples
        self.past = np.zeros((self.delay_time + 1, sample_size))

    def apply(self, sound):
        # Ensure sound is a 2D array with shape (num_samples, sample_size)
        if sound.ndim != 2 or sound.shape[1] != self.sample_size:
            raise ValueError(
                "Input sound must be a 2D array with shape (num_samples, sample_size)")

        # Create an output array
        output = np.zeros_like(sound)

        for i in range(sound.shape[0]):
            # Get the current sample
            current_sample = sound[i]

            # Get the delayed sample from the past buffer
            if i >= self.delay_time:
                delayed_sample = self.past[-self.delay_time]
            else:
                delayed_sample = np.zeros(self.sample_size)

            # Calculate the echo effect
            echo_sample = (current_sample + delayed_sample *
                           self.feedback) * self.wet_dry_mix

            # Store the current sample in the past buffer
            self.past = np.roll(self.past, shift=-1, axis=0)
            self.past[-1] = current_sample

            # Combine the original sound and the echo
            output[i] = current_sample * (1 - self.wet_dry_mix) + echo_sample

        return output


class Synth2():
    def __init__(self, fourier_nn, stdout: Queue = None, port_name=None):
        self.stdout = stdout
        self.live_synth: Process = None
        self.notes_ready = False
        self.fourier_nn: FourierNN = fourier_nn
        self.fs = 44100  # Sample rate
        self.max_parralel_notes = 3
        self.current_frame = 0
        t = np.array([
            utils.midi_to_freq(f) *
            np.linspace(0, 2*np.pi, self.fs)
            for f in range(128)
        ])
        self.t_buffer = torch.tensor(t, dtype=torch.float32)
        print(self.t_buffer.shape)
        self.port_name = port_name

    def set_port_name(self, port_name):
        self.port_name = port_name

    def play_init_sound(self):
        f1 = 440  # Frequency of the "du" sound (in Hz)
        f2 = 660  # Frequency of the "dl" sound (in Hz)
        f3 = 880  # Frequency of the "de" sound (in Hz)
        f4 = 1320  # Frequency of the "dii" sound (in Hz)
        t1 = 0.25  # Duration of the "du" sound (in seconds)
        t2 = 0.25  # Duration of the "dl" sound (in seconds)
        t3 = 0.25  # Duration of the "de" sound (in seconds)
        t4 = 0.4  # Duration of the "dii" sound (in seconds)

        # Generate the "duuu" sound
        t = np.arange(int(t1 * self.fs)) / self.fs
        sound1 = 0.5 * np.sin(2 * np.pi * f1 * t)

        # Generate the "dl" sound
        t = np.arange(int(t2 * self.fs)) / self.fs
        sound2 = 0.5 * np.sin(2 * np.pi * f2 * t)

        # Generate the "diii" sound
        t = np.arange(int(t3 * self.fs)) / self.fs
        sound3 = 0.5 * np.sin(2 * np.pi * f3 * t)

        # Generate the "dub" sound
        t = np.arange(int(t4 * self.fs)) / self.fs
        sound4 = 0.5 * np.sin(2 * np.pi * f4 * t)

        # Concatenate the sounds to form "duuudldiiidub"
        audio = np.concatenate([sound1, sound2, sound3, sound4])
        output = np.array(
            audio * 32767 / np.max(np.abs(audio)) / 2).astype(np.int16)
        sd.play(output, blocking=True)
        print("Ready")

    def live_synth_loop(self):
        print("Live Synth is running")

        if not self.port_name:
            self.port_name = mido.get_input_names()[0]

        p = pyaudio.PyAudio()
        CHUNK = 2048  # Increased chunk size
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        frames_per_buffer=CHUNK,
                        rate=self.fs,
                        output=True)
        current_frame = 0
        cycle_frame = 0
        self.adsr_envelope = ADSR(
            0.1,
            0.1,
            0.75,
            0.2,
            self.fs)

        notes = {}
        model = self.fourier_nn.current_model.to(self.fourier_nn.device)
        self.play_init_sound()
        with mido.open_input(self.port_name) as midi_input:
            while True:
                # for _ in utils.timed_loop(True):
                available_buffer = stream.get_write_available()
                if available_buffer == 0:
                    continue
                midi_event = midi_input.poll()
                if midi_event:
                    print(midi_event)
                    # midi_event.type
                    # midi_event.note
                    # midi_event.velocity
                    if midi_event.type == 'note_on':  # Note on
                        print("Note on",
                              midi_event.note,
                              utils.midi_to_freq(midi_event.note))

                        notes[midi_event.note] = [True, midi_event.velocity]
                        self.adsr_envelope.reset_note(midi_event.note)
                    elif midi_event.type == 'note_off':  # Note off
                        print("Note off",
                              midi_event.note,
                              utils.midi_to_freq(midi_event.note))
                        if midi_event.note in notes:
                            # del notes[midi_event.note]
                            notes[midi_event.note][0] = False

                if len(notes) > 0:
                    # y = torch.zeros(size=(len(notes), available_buffer),
                    #                 device=self.fourier_nn.device)
                    y = np.zeros((len(notes), available_buffer),
                                 dtype=np.float32)
                    to_delete = []
                    for i, (note, data) in enumerate(notes.items()):
                        with torch.no_grad():
                            x = utils.wrap_concat(
                                self.t_buffer[note], cycle_frame, cycle_frame + available_buffer).to(self.fourier_nn.device)
                            if data[0]:
                                envelope = self.adsr_envelope.get_ads_envelope(
                                    note, available_buffer)
                            else:
                                envelope = self.adsr_envelope.get_r_envelope(
                                    note, available_buffer)
                                if self.adsr_envelope.has_note_ended(note):
                                    to_delete.append(note)

                            model_output = model(x.unsqueeze(1))
                            y[i, :] = model_output.cpu().numpy().astype(
                                np.float32) * envelope
                            # y[i, :] = model(
                            #     x.unsqueeze(1)).cpu().numpy().astype(np.float32)  # * envelope  # * (amplitude/127)
                            # y[i, :] = np.sin(
                            #     x.numpy().astype(np.float32)) * envelope

                    for note in to_delete:
                        del notes[note]

                    audio_data = sum_signals(y)
                    # print(np.max(np.abs(audio_data)))
                    audio_data = normalize(audio_data)
                    audio_data = np.clip(audio_data, -1, 1)
                    audio_data *= 0.7
                    stream.write(audio_data.astype(np.float32),
                                 available_buffer,
                                 exception_on_underflow=True)
                # if cycle_frame <= 100:
                #     print(f"available_buffer: {available_buffer}")
                current_frame = (current_frame + available_buffer)
                cycle_frame = (cycle_frame + available_buffer) % self.fs

    def run_live_synth(self):
        if not self.live_synth:
            print("spawning live synth")
            self.live_synth = Process(target=self.live_synth_loop)
            self.live_synth.start()
        else:
            print("killing live synth")
            self.live_synth.terminate()
            self.live_synth.join()
            self.live_synth = None
            print("live synth killed")
        atexit.register(utils.DIE, self.live_synth, 0, 0)

    def __getstate__(self) -> object:
        live_synth = self.live_synth
        del self.live_synth
        Synth_dict = self.__dict__.copy()
        self.live_synth = live_synth
        return Synth_dict

    def __setstate__(self, state):
        # Load the model from a file after deserialization
        self.__dict__.update(state)
        if self.stdout is not None:
            if current_process().name != 'MainProcess':
                sys.stdout = utils.QueueSTD_OUT(queue=self.stdout)


def normalize(signal):
    max_val = np.max(np.abs(signal))
    if np.isnan(max_val) or max_val == 0:
        return signal
    return signal / np.max(np.abs(signal))


def rms(signal):
    return np.sqrt(np.mean(signal**2))


def mix_signals(signal1, signal2):
    # Normalize both signals
    signal1 = normalize(signal1)
    signal2 = normalize(signal2)

    # Calculate RMS levels
    rms1 = rms(signal1)
    rms2 = rms(signal2)

    # Adjust gain
    gain1 = 1 / rms1
    gain2 = 1 / rms2

    # Apply gain
    signal1 *= gain1
    signal2 *= gain2

    # Mix signals
    mixed_signal = signal1 + signal2

    # Normalize mixed signal to prevent clipping
    mixed_signal = normalize(mixed_signal)

    return mixed_signal


def sum_signals(signals):
    # Normalize each signal
    normalized_signals = np.array([normalize(signal) for signal in signals])

    # Sum the signals along dimension 0
    summed_signal = np.sum(normalized_signals, axis=0)

    # Normalize the combined signal to prevent clipping
    summed_signal = normalize(summed_signal)

    return summed_signal


def normalize_torch(signal):
    return signal / torch.max(torch.abs(signal))


def sum_signals_torch(signals):
    # Convert signals to a PyTorch tensor and move to GPU
    # signals = torch.stack([signals
    #                       for signal in signals])

    # Normalize each signal
    normalized_signals = torch.stack(
        [sum_signals_torch(signal) for signal in signals])

    # Sum the signals along dimension 0
    summed_signal = torch.sum(normalized_signals, dim=0)

    # Normalize the combined signal to prevent clipping
    summed_signal = normalize(summed_signal)

    return summed_signal
