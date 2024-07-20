import time
import mido
import pyaudio
import numpy as np
import threading


class AudioPlayer:
    def __init__(self, audio_sample, channels=1, rate=44100, format=pyaudio.paFloat32):
        self.audio_sample = audio_sample
        self.channels = channels
        self.rate = rate
        self.format = format
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.format,
                                  channels=self.channels,
                                  rate=self.rate,
                                  output=True)
        self.playing = False

    def play(self, loop=False):
        self.playing = True
        self.loop = loop
        self.thread = threading.Thread(target=self.stream_audio)
        self.thread.daemon = True
        self.thread.start()

    def stream_audio(self):
        chunk_size = 1024  # Number of samples to write at once
        chunks = np.array_split(self.audio_sample, len(
            self.audio_sample) // chunk_size)
        chunk_index = 0
        while self.playing:
            self.stream.write(chunks[chunk_index].tobytes())
            chunk_index += 1
            if self.loop and chunk_index == len(chunks):
                chunk_index = 0
            elif chunk_index == len(chunks):
                self.playing = False
                break

    def pause(self):
        self.playing = False
        self.thread.join()

    def resume(self):
        self.play(loop=self.loop)

    def stop(self):
        self.playing = False
        self.thread.join()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def wait(self):
        while self.playing:
            time.sleep(0.1)  # Sleep for a short time to prevent busy waiting


def a():
    sample_rate = 44100  # in Hz
    freq = 220.0    # sine frequency, Hz
    duration = 1.0   # seconds

    # Generate the time values for each sample
    t = np.linspace(0, duration, int(sample_rate * duration), False)

    # Generate a sine wave
    audio_sample = np.sin(freq * t * 2 * np.pi)
    audio_sample = (audio_sample * 32767 / np.max(np.abs(audio_sample))) / 2

    player = AudioPlayer(audio_sample, rate=sample_rate)
    timestamp = time.perf_counter_ns()
    player.play()
    res = time.perf_counter_ns()
    player.wait()
    player.stop()
    print(f"time taken: {(res-timestamp)/1_000_000_000}s")


def b():
    duration = 1
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    notes = [(np.sin(freq * t * 2 * np.pi), print(freq))[0] for freq in (440.0 *
                                                                         (2.0 ** ((midi_note - 69) / 12.0)) for midi_note in range(128))]
    playing_notes: dict[str, AudioPlayer] = {}

    midiin = mido.open_input('LoopBe Internal MIDI 0')
    try:
        while True:
            msg = midiin.receive()
            print(msg)
            if 'note_on' == msg.type:
                if msg.note not in playing_notes:
                    playing_notes[msg.note] = AudioPlayer(notes[msg.note])
                playing_notes[msg.note].play(loop=True)
            if 'note_off' == msg.type:
                playing_notes[msg.note].pause()
    except KeyboardInterrupt:
        for k, v in playing_notes.keys():
            v.stop()


b()
