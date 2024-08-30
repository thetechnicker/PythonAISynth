from time import sleep
import numpy as np
import sounddevice as sd

# Constants
fs = 44100  # Sample rate
T = 5.0    # seconds

# Generate time array
t = np.arange(fs * T) / fs

# Generate the sine waves
sine_wave_1 = np.sin(2 * np.pi * 440 * t).reshape(-1, 1)
sine_wave_2 = np.sin(2 * np.pi * 500 * t).reshape(-1, 1)

# Define your callback functions


def callback1(outdata, frames, time, status):
    # if status:
    #     print(status)
    # this will play the first sine wave
    # print(len(outdata))
    outdata[:] = sine_wave_1[:frames]


def callback2(outdata, frames, time, status):
    # if status:
    #     print(status)
    # this will play the second sine wave
    outdata[:] = sine_wave_2[:frames]


# Create streams
stream1 = sd.OutputStream(
    callback=callback1, samplerate=fs, channels=1, dtype=np.float32)
stream2 = sd.OutputStream(
    callback=callback2, samplerate=fs, channels=1, dtype=np.float32)

# Start the streams
stream1.start()
stream2.start()

sleep(5)

# Stop the streams
stream1.stop()
stream2.stop()
