import pyaudio
import numpy as np

# Create a PyAudio object
p = pyaudio.PyAudio()

# Define the audio stream parameters
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# Open the input and output streams
stream_in = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)

stream_out = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)

print("Sending and receiving data...")

# Send data to the input of the stream and compare it to the output
for _ in range(int(RATE / CHUNK * RECORD_SECONDS)):
    # Generate a random chunk of data
    data_out = np.random.rand(CHUNK).astype(np.float32).tobytes()

    # Send the data to the output stream
    stream_out.write(data_out)

    # Read the same amount of data from the input stream
    data_in = stream_in.read(CHUNK)

    # Compare the input data to the output data
    if data_in != data_out:
        print("Data mismatch!")
else:
    print("Data matched for all chunks")

# Close the input and output streams
stream_in.stop_stream()
stream_in.close()
stream_out.stop_stream()
stream_out.close()

# Terminate the PyAudio object
p.terminate()
