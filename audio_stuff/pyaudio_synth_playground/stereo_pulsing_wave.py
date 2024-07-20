import pyaudio
import numpy as np

# Choose your settings
sample_rate = 44100  # in Hz
freq = 440.0    # sine frequency, Hz
mod_freq = 0.25  # modulation frequency, Hz
duration = 5.0   # seconds

# Generate the time values for each sample
t = np.linspace(0, duration, int(sample_rate * duration), False)

# Generate a sine wave
note = np.sin(freq * t * 2 * np.pi)

# Generate two slower sine waves to modulate the amplitude
modulator_left = np.sin(mod_freq * t * 2 * np.pi)
# offset by pi for opposite phase
modulator_right = np.sin(mod_freq * t * 2 * np.pi + np.pi)

# Modulate the amplitude
note_modulated_left = note * modulator_left
note_modulated_right = note * modulator_right

# Ensure that highest value is in 16-bit range
audio_left = note_modulated_left * \
    (2**15 - 1) / np.max(np.abs(note_modulated_left))
audio_right = note_modulated_right * \
    (2**15 - 1) / np.max(np.abs(note_modulated_right))

# Convert to 16-bit data
audio_left = audio_left.astype(np.int16)
audio_right = audio_right.astype(np.int16)

# Interleave left and right audio channels
audio_stereo = np.vstack((audio_left, audio_right)).T.flatten()

# Open a new PyAudio stream and play the audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=2,
                rate=sample_rate,
                output=True)

stream.write(audio_stereo.tobytes())
stream.stop_stream()
stream.close()

# Terminate the PyAudio session
p.terminate()
