import time
import numpy as np
import pygame
import pygame.midi
import sounddevice as sd
import threading

# Define the sample rate
fs = 44100  # Sample rate


def rescale_audio(audio):
    # Find the maximum absolute value in the audio
    max_val = max(abs(audio.min()), audio.max())

    # If max_val is zero, it means the audio is silent. In this case, we return the audio as it is.
    if max_val == 0:
        return audio

    # Rescale the audio to fit within the range -1 to 1
    return audio / max_val


class myBuffer:
    def __init__(self) -> None:
        self._dict = {}
        self.lock = threading.Lock()

    def add_sample(self, name: str, buffer: np.ndarray, repeat: bool = False):
        with self.lock:
            self._dict[name] = (buffer, repeat)
            max_len = max(len(i[0]) for i in self._dict.values())

            for name, (i, repeat) in self._dict.items():
                zero_padding = np.zeros(max_len - len(i))
                self._dict[name] = (np.concatenate(
                    (i, zero_padding)), repeat)

    def remove_sample(self, name: str):
        with self.lock:
            del self._dict[name]

    def __getitem__(self, index):
        result = None
        with self.lock:
            if len(self._dict) > 0:
                max_len = max(len(i[0]) for i in self._dict.values())

                for name, (i, repeat) in self._dict.items():
                    if len(i) == max_len:
                        continue
                    zero_padding = np.zeros(max_len - len(i))
                    self._dict[name] = (np.concatenate(
                        (i, zero_padding)), repeat)

                result = np.sum([i[0]
                                for i in self._dict.values()], 0).reshape(-1, 1)

                if isinstance(index, slice):
                    start, stop, step = index.indices(max_len)
                    result_range = range(start, stop, step)
                    if len(result) < index.stop:
                        zero_padding = np.zeros((index.stop - len(result), 1))
                        result = np.concatenate((result, zero_padding))
                    to_delet = []
                    for name, (i, repeat) in self._dict.items():
                        deleted_part = np.take(i, result_range)
                        self._dict[name] = (np.delete(i, result_range), repeat)
                        if repeat:
                            self._dict[name] = (np.concatenate(
                                (self._dict[name][0], deleted_part)), repeat)
                        if len(self._dict[name][0]) == 0:
                            to_delet.append(name)
                    for name in to_delet:
                        del self._dict[name]
                    return result[index]
                else:
                    try:
                        result = result[index]
                        for name, (i, repeat) in self._dict.items():
                            deleted_part = i[index]
                            self._dict[name] = (np.delete(i, index), repeat)
                            if repeat:
                                self._dict[name] = (np.concatenate(
                                    (self._dict[name][0], [deleted_part])), repeat)
                        return result
                    except IndexError:
                        return 0
            else:
                if isinstance(index, slice):
                    return np.zeros((len(range(index.start or 0, index.stop, index.step or 1)), 1))
                else:
                    return np.zeros((1, 1))


buffer = myBuffer()


def callback(outdata, frames, _time, status):
    outdata[:] = rescale_audio(buffer[:frames])


stream = sd.OutputStream(callback=callback, channels=1, samplerate=fs)

# Start the stream
stream.start()
# Initialize the pygame.midi module
pygame.midi.init()

# Get the default input device ID
input_id = pygame.midi.get_default_input_id()

# Create an Input object
midi_input = pygame.midi.Input(input_id)
playing_notes = {}
try:
    while True:
        if midi_input.poll():
            # If there is data waiting to be read from the input device
            midi_events = midi_input.read(10)

            # Process the MIDI events
            for event in midi_events:
                data, timestamp = event
                print(data)

                # The data is a list of values sent by the MIDI device.
                # The first value is the status byte, which tells us what kind of event this is.
                # The following values are parameters for the event.
                status = data[0]
                parameters = data[1:]

                # For example, if this is a Note On event, generate a sound
                if status == 144:  # Note On event
                    # The first parameter is the note number
                    note = parameters[0]
                    # The second parameter is the velocity
                    velocity = parameters[1]

                    # Generate a sound based on the note and velocity
                    # Convert MIDI note number to frequency
                    f = 440 * (2 ** ((note - 69) / 12))
                    t = np.arange(fs) / fs
                    # Scale volume by velocity
                    sample = np.sin(2 * np.pi * f * t) * (velocity / 127)
                    buffer.add_sample(note, sample, False)
                elif status == 128:
                    # The first parameter is the note number
                    note = parameters[0]
                    # The second parameter is the velocity
                    velocity = parameters[1]
                    try:
                        buffer.remove_sample(note)
                    except KeyError as e:
                        print(buffer._dict.keys(), e)

        time.sleep(0.01)  # Sleep for a bit to reduce CPU usage
except KeyboardInterrupt:
    pass
# Close the MIDI input
midi_input.close()
# Stop the stream
stream.stop()
