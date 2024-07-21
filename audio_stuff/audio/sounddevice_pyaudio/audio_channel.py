import pyaudio
import numpy as np


# Create a new PyAudio object
p = pyaudio.PyAudio()

# The callback function that will be called when audio is available


def callback(in_data, frame_count, time_info, status):
    # Convert the input data to a NumPy array
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    # Apply volume adjustment (this is a very simple example, you might want to use a more sophisticated method)
    # audio_data *= volume_level.get()
    # Convert the NumPy array back to bytes
    out_data = audio_data.tobytes()

    return (out_data, pyaudio.paContinue)


# Open a new audio stream
stream = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=44100,
                input=True,
                output=True,
                stream_callback=callback)

# Start the audio stream
stream.start_stream()

# Keep the script running while the audio stream is open
while stream.is_active():
    # You can update the volume level here based on user input
    pass

# Stop the audio stream
stream.stop_stream()
stream.close()

# Terminate the PyAudio object
p.terminate()
