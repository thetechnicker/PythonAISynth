import sounddevice as sd
import numpy as np

# Erstellen Sie ein Sinussignal
fs = 44100  # Abtastrate
f = 440  # Die Frequenz des Signals
seconds = 5  # Die Dauer des Signals
t = np.arange(fs * seconds) / fs
signal = 0.5 * np.sin(2 * np.pi * f * t)

# Wählen Sie das virtuelle Gerät aus
device = '<MySink>'  # Ersetzen Sie dies durch den Namen Ihres virtuellen Geräts

# Spielen Sie das Signal ab
sd.play(signal, samplerate=fs, device=device)

# Warten Sie, bis die Wiedergabe beendet ist
sd.wait()
