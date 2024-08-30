import pyaudio
import numpy as np

# Choose your settings
sample_rate = 44100  # in Hz
freq = 440.0    # sine frequency, Hz
mod_freq = 0.5  # modulation frequency, Hz
duration = 5.0   # seconds

# Generate the time values for each sample
t = np.linspace(0, duration, int(sample_rate * duration), False)

# Generate a sine wave
note = np.sin(freq * t * 2 * np.pi)

# Generate a slower sine wave to modulate the amplitude
modulator = np.sin(mod_freq * t * 2 * np.pi)

# Modulate the amplitude
note_modulated = note * modulator

# Ensure that highest value is in 16-bit range
audio = note_modulated * (2**15 - 1) / np.max(np.abs(note_modulated))
audio = audio.astype(np.int16)

# Open a new PyAudio stream and play the audio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
                channels=1,
                rate=sample_rate,
                output=True)
stream.write(audio.tobytes())
stream.stop_stream()
stream.close()

# Terminate the PyAudio session
p.terminate()
