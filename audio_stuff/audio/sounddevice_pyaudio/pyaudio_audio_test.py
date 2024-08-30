import pyaudio
import numpy as np
import multiprocessing
import time

# Define the audio properties
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

# Create a PyAudio instance
p = pyaudio.PyAudio()


class Mixer:
    def __init__(self, num_channels=1) -> None:
        self.channels = [Channel() for i in range(num_channels)]

    def get_channel(self, id):
        if id < len(self.channels):
            return self.channels[id]
        raise ValueError(f"value {id} out of range")

    def stop(self):
        for channel in self.channels:
            channel.stop()


class Channel:
    def __init__(self):
        self.q = multiprocessing.Queue()
        self.p = multiprocessing.Process(target=self.play_audio)
        self.p.start()

    def play_audio(self):
        # Open a new audio stream
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True)

        # Play audio in a loop until a stop command is received
        loop = True
        while loop:
            # Get the next command and sound data from the queue
            command, sound_data = self.q.get()

            # If the command is the stop command, break the loop
            if command == 'stop':
                break

            # If the command is pause, wait for resume command
            elif command == 'pause':
                while True:
                    new_command, _ = self.q.get()
                    if new_command == 'resume':
                        break

            # If the command is play, play the audio sample once
            elif command == 'play':
                stream.write(sound_data)

            # If the command is play_loop, play the audio sample in a loop
            elif command == 'play_loop':
                while True:
                    stream.write(sound_data)
                    if not self.q.empty():
                        new_command, _ = self.q.get()
                        if new_command == 'pause':
                            while True:
                                new_new_command, _ = self.q.get()
                                if new_new_command == 'resume':
                                    break
                        if new_command == 'stop':
                            loop = False
                            break

        # Close the audio stream
        stream.stop_stream()
        stream.close()

    def play(self, sound_data):
        self.q.put(('play', sound_data))

    def play_loop(self, sound_data):
        self.q.put(('play_loop', sound_data))

    def pause(self):
        self.q.put(('pause', None))

    def resume(self):
        self.q.put(('resume', None))

    def stop(self):
        self.q.put(('stop', None))
        self.p.join(2)
        self.p.terminate()


if __name__ == "__main__":
    num_channels = 1
    mixer = Mixer(num_channels=num_channels)

    for i in range(num_channels):
        sample = (np.sin(2*np.pi*np.arange(RATE*1)*(440+20*i)/RATE)
                  ).astype(np.float32).tobytes()
        mixer.get_channel(i).play_loop(sample)
        time.sleep(1)

    time.sleep(10)
    mixer.stop()
